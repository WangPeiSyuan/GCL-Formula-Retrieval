import torch
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from GCL.models import SingleBranchContrast
from torch_geometric.nn import  GINConv, global_add_pool
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
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GConv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(make_gin_conv(input_dim, hidden_dim))
            else:
                self.layers.append(make_gin_conv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv, bn in zip(self.layers, self.batch_norms):
            z = conv(z, edge_index)
            z = self.activation(z)
            z = bn(z)
            zs.append(z)
        gs = [global_add_pool(z, batch) for z in zs]
        z, g = [torch.cat(x, dim=1) for x in [zs, gs]]
        return z, g


class FC(nn.Module):
    def __init__(self, hidden_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)


class Encoder(torch.nn.Module):
    def __init__(self, encoder, local_fc, global_fc):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.local_fc = local_fc
        self.global_fc = global_fc

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

    def project(self, z, g):
        return self.local_fc(z), self.global_fc(g)

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
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

        z, g = encoder_model(data.x, data.edge_index, data.batch)
        z, g = encoder_model.project(z, g)
        loss = contrast_model(h=z, g=g, batch=data.batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        global_steps += 1
    return epoch_loss, global_steps

def get_embedding(encoder_model, dataloader):
    emb = {}
    encoder_model.eval()
    for data in dataloader:
        data = data.to(device_name)
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g = encoder_model(data.x, data.edge_index, data.batch)
        batch_detatch(data.y, g.detach().cpu().numpy(), emb)
    return emb

def test(encoder_model, query_dataloader, train_dataloader, epoch, run_id, encode, model, batch_size):
    query_emb = get_embedding(encoder_model, query_dataloader)
    emb_dict = get_embedding(encoder_model, train_dataloader)
    result = FormulaRetrieval(emb_dict, query_emb, 1000)
    result.create_retrieval_file(encode, model, batch_size, epoch=epoch, run_id=run_id)


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
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--encode", type=str, default='slt')
    parser.add_argument("--pretrained", default=False, action='store_true')
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--run_id", type=str, required=True)
    # parser.add_argument("--seed", type=int, default=20)
    
    args = vars(parser.parse_args())
    batch_size = args['bs']
    encode = args['encode']
    pretrained = args['pretrained']
    lr = args['lr']
    run_id = args['run_id']
    # seed = args['seed']
    epochs = args['epoch']
    # setup_seed(seed)
    if pretrained:
        input_dim = 200
        hidden_dim = 256
    else:
        input_dim = 2
        hidden_dim = 32

    train_dataset = Equation(encode=encode, pretrained=pretrained)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    query_dataset = Equation(encode=encode, training=False, pretrained=pretrained)
    query_dataloader = DataLoader(query_dataset, batch_size=20) 
    judge_dataset = Equation(encode=encode, training=False, judge=True, pretrained=pretrained)
    judge_dataloader = DataLoader(judge_dataset, batch_size=256)
    device = torch.device(device_name)

    gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=2).to(device)
    fc1 = FC(hidden_dim=hidden_dim*2)
    fc2 = FC(hidden_dim=hidden_dim*2)
    encoder_model = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=lr)


    loss_list = []
    max_steps = epochs*len(train_dataloader)
    global_steps = 0
    base_lr = lr


    for epoch in range(1, epochs+1):

        lr = adjust_learning_rate(optimizer, base_lr, 0.0001, global_steps, max_steps)
        loss, global_steps = train(encoder_model, contrast_model, train_dataloader, optimizer, global_steps)
        loss_list.append(loss)
        print('Epoch {}: \t Loss: {}'.format(epoch, loss))
        
    # test(encoder_model, query_dataloader, train_dataloader, str(epochs), run_id, encode, 'InfoG', batch_size)
    test(encoder_model, query_dataloader, judge_dataloader, 'end', run_id, encode, 'InfoG', batch_size)    

    file_path = "Retrieval_result/InfoG/"+encode+"/"+str(batch_size)+"/"+str(run_id)
    torch.save(encoder_model.state_dict(), file_path+"/model")


if __name__ == '__main__':
    main()

