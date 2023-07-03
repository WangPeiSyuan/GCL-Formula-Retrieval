import torch
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader

from copy import deepcopy
import random
import argparse 
import math
import numpy as np

from FormulaRetrieval import FormulaRetrieval
from EquationData import Equation


device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_gin_conv(input_dim, out_dim):
    return GINConv(nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)))

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        project_dim = hidden_dim * num_layers
        self.project = torch.nn.Sequential(
            nn.Linear(project_dim, project_dim),
            nn.ReLU(inplace=True),
            nn.Linear(project_dim, project_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = F.relu(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
    
        return z, g


class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor

    def forward(self, x, edge_index, batch):
        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index)
        z, g = self.encoder(x, edge_index, batch)
        z1, g1 = self.encoder(x1, edge_index1, batch)
        z2, g2 = self.encoder(x2, edge_index2, batch)
        return z, g, z1, z2, g1, g2

def batch_detatch(batch_index, embs, emb_dict):
    tmp_dict = dict(zip(batch_index, embs))
    emb_dict.update(tmp_dict)

def train(encoder_model, contrast_model, dataloader, optimizer, global_steps):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data = data.to(device_name)
        optimizer.zero_grad()

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32)
        _, _, _, _, g1, g2 = encoder_model(data.x, data.edge_index, data.batch)
        # g1, g2 = [encoder_model.encoder.project(g) for g in [g1, g2]]
        loss = contrast_model(g1=g1, g2=g2, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        global_steps += 1
    return g1, g2, epoch_loss, global_steps

def test(encoder_model, query_dataloader, train_dataloader, epoch, run_id, encode, model, batch_size):
    query_emb = get_embedding(encoder_model, query_dataloader)
    emb_dict = get_embedding(encoder_model, train_dataloader)
    result = FormulaRetrieval(emb_dict, query_emb, 1000)
    result.create_retrieval_file(encode, model, batch_size, epoch=epoch, run_id=run_id)


def get_embedding(encoder_model, dataloader):
    emb = {}
    encoder_model.eval()
    with torch.no_grad():    
        for data in dataloader:
            data = data.to(device_name)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            _, g, _, _, _, _ = encoder_model(data.x, data.edge_index, data.batch)
            batch_detatch(data.y, g.detach().cpu().numpy(), emb)

    return emb


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    
def adjust_learning_rate(optimizer, base_lr, end_lr, step, max_steps):
    q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
    lr = base_lr * q + end_lr * (1 - q)
    set_lr(optimizer, lr)
    return lr

def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=2048)
    parser.add_argument("--encode", type=str, default='slt')
    parser.add_argument("--pretrained", default=False, action='store_true')
    parser.add_argument("--lr", type=int, default=0.01)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--aug_p", type=int, default=0.1)
    parser.add_argument("--aug_id", type=int, default=5)
    
    args = vars(parser.parse_args())
    batch_size = args['bs']
    encode = args['encode']
    pretrained = args['pretrained']
    lr = args['lr']
    run_id = args['run_id']
    seed = args['seed']
    aug_p = args['aug_p']
    aug_id = args['aug_id']
    epochs = args['epoch']

    # setup_seed(seed)
    train_dataset = Equation(encode=encode, pretrained=pretrained)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    query_dataset = Equation(encode=encode, training=False, pretrained=pretrained)
    query_dataloader = DataLoader(query_dataset, batch_size=20) 
    judge_dataset = Equation(encode=encode, training=False, judge=True, pretrained=pretrained)
    judge_dataloader = DataLoader(judge_dataset, batch_size=256)
    device = torch.device(device_name)

    aug = {1: A.FeatureMasking(pf=0.3), 
        2: A.EdgeRemoving(pe=0.3), 
        3: A.NodeDropping(pn=0.3),
        4: A.EdgeAttrMasking(pf=0.3),
        5: A.RandomChoice([ A.NodeDropping(pn=aug_p),
                            A.FeatureMasking(pf=aug_p),
                            A.EdgeAttrMasking(pf=aug_p),
                            A.EdgeRemoving(pe=aug_p)], 1)
    }

    aug1 = A.Identity()
    aug2 = aug[aug_id]
    if pretrained:
        input_dim = 200
        hidden_dim = 128
    else:
        input_dim = 2
        hidden_dim = 32

    gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2)).to(device)
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.03), mode='G2G').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=lr)
    loss_list=[]

    max_steps = epochs*len(train_dataloader)
    global_steps = 0
    base_lr = lr


    for epoch in range(1, epochs+1):

        # lr = adjust_learning_rate(optimizer, base_lr, 0.0001, global_steps, max_steps)
        g1, g2, loss, global_steps = train(encoder_model, contrast_model, train_dataloader, optimizer, global_steps)
        print('Epoch {}: \t Loss: {}'.format(epoch, loss))
        loss_list.append(loss)
    
    # test(encoder_model, query_dataloader, train_dataloader, str(epochs), run_id, encode, 'GCL', batch_size)
    test(encoder_model, query_dataloader, judge_dataloader, 'end', run_id, encode, 'GCL', batch_size)
    
    file_path = "Retrieval_result/GCL/"+encode+"/"+str(batch_size)+"/"+str(run_id)
    torch.save(encoder_model.state_dict(), file_path+"/model")

if __name__=="__main__":
    main()
