import torch
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from GCL.models import BootstrapContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader

from copy import deepcopy 
import random
import argparse 
import math
import numpy as np

from FormulaRetrieval import FormulaRetrieval
from EquationData import Equation

class Normalize(torch.nn.Module):
    def __init__(self, dim=None, norm='batch'):
        super().__init__()
        if dim is None or norm == 'none':
            self.norm = lambda x: x
        if norm == 'batch':
            self.norm = torch.nn.BatchNorm1d(dim)
        elif norm == 'layer':
            self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


def make_gin_conv(input_dim: int, out_dim: int) -> GINConv:
    mlp = torch.nn.Sequential(
        torch.nn.Linear(input_dim, out_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(out_dim, out_dim))
    return GINConv(mlp)


class GConv(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2,
                 encoder_norm='batch', projector_norm='batch'):
        super(GConv, self).__init__()
        self.activation = torch.nn.PReLU()
        self.dropout = dropout

        self.layers = torch.nn.ModuleList()
        self.layers.append(make_gin_conv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(make_gin_conv(hidden_dim, hidden_dim))

        self.batch_norm = Normalize(hidden_dim, norm=encoder_norm)
        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=projector_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
            z = F.dropout(z, p=self.dropout, training=self.training)
        z = self.batch_norm(z)
        return z, self.projection_head(z)
        
class Encoder(torch.nn.Module):
    def __init__(self, encoder, augmentor, hidden_dim, dropout=0.2, predictor_norm='batch'):
        super(Encoder, self).__init__()
        self.online_encoder = encoder
        self.target_encoder = None
        self.augmentor = augmentor
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            Normalize(hidden_dim, norm=predictor_norm),
            torch.nn.PReLU(),
            torch.nn.Dropout(dropout))

    def get_target_encoder(self):
        if self.target_encoder is None:
            self.target_encoder = deepcopy(self.online_encoder)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
        return self.target_encoder

    @torch.no_grad()
    def update_target_encoder(self, mm: float):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        aug1, aug2 = self.augmentor
       

        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)


        h, h_online = self.online_encoder(x, edge_index, edge_weight)
        h1, h1_online = self.online_encoder(x1, edge_index1, edge_weight1)
        h2, h2_online = self.online_encoder(x2, edge_index2, edge_weight2)

        g = global_add_pool(h, batch)
        g1 = global_add_pool(h1, batch)
        h1_pred = self.predictor(h1_online)
        g2 = global_add_pool(h2, batch)
        h2_pred = self.predictor(h2_online)
        
        with torch.no_grad():
            _, h1_target = self.get_target_encoder()(x1, edge_index1, edge_weight1)
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)
            g1_target = global_add_pool(h1_target, batch)
            g2_target = global_add_pool(h2_target, batch)

        return g1, g2, h1_pred, h2_pred, h1_target, h2_target, g1_target, g2_target

def batch_detatch(batch_index, embs, emb_dict):
    tmp_dict = dict(zip(batch_index, embs))
    emb_dict.update(tmp_dict)

def train(encoder_model, contrast_model, dataloader, optimizer, global_steps):
    encoder_model.train()
    total_loss = 0

    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32).to(data.batch.device)

        optimizer.zero_grad()
        _, _, h1_pred, h2_pred, h1_target, h2_target, _, _ = encoder_model(data.x, data.edge_index, batch=data.batch)
        loss = contrast_model(h1_pred=h1_pred, h2_pred=h2_pred, h1_target=h1_target.detach(), h2_target=h2_target.detach())

        loss.backward()
        optimizer.step()
        encoder_model.update_target_encoder(0.99)

        total_loss += loss.item()
        global_steps+=1

    return total_loss, global_steps
    
def test(encoder_model, query_dataloader, train_dataloader, epoch, run_id, encode, model, batch_size):
    query_emb = get_embedding(encoder_model, query_dataloader)
    emb_dict = get_embedding(encoder_model, train_dataloader)
    result = FormulaRetrieval(emb_dict, query_emb, 1000)
    result.create_retrieval_file(encode, model, batch_size, epoch=epoch, run_id=run_id)


def get_embedding(encoder_model, dataloader):
    encoder_model.eval()
    emb = {}
    with torch.no_grad():    
        for data in dataloader:
    #         print(graph_id)
            data = data.to('cuda')
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            g1, g2, _, _, _, _, _, _ = encoder_model(data.x, data.edge_index, batch=data.batch)
            z = torch.cat([g1, g2], dim=1)
            batch_detatch(data.y, z.detach().cpu().numpy(), emb)
    # x = torch.cat(x, dim=0)

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
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--encode", type=str, default='slt')
    parser.add_argument("--pretrained", default=False, action='store_true')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--run_id", type=str, required=True)
    # parser.add_argument("--seed", type=int, default=-1)
    
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
    device = torch.device('cuda')

    aug1 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.1)])
    gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2).to(device)
    encoder_model = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=hidden_dim).to(device)
    contrast_model = BootstrapContrast(loss=L.BootstrapLatent(), mode='L2L').to(device)
    optimizer = Adam(encoder_model.parameters(), lr=lr)

    max_steps = epochs*len(train_dataloader)
    global_steps = 0
    base_lr = lr

    for epoch in range(1, epochs+1):

        lr = adjust_learning_rate(optimizer, base_lr, 0.001, global_steps, max_steps)
        loss, global_steps = train(encoder_model, contrast_model, train_dataloader, optimizer, global_steps)
        print('Epoch {}: \tLoss:'.format(epoch))
        
    # test(encoder_model, query_dataloader, train_dataloader, str(epochs), run_id, encode, 'BGRL', batch_size)
    test(encoder_model, query_dataloader, judge_dataloader, 'end', run_id, encode, 'BGRL', batch_size)
    
    file_path = "Retrieval_result/BGRL/"+encode+"/"+str(batch_size)+"/"+str(run_id)
    torch.save(encoder_model.state_dict(), file_path+"/model")

if __name__=="__main__":
    main()
