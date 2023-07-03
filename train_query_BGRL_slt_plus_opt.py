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
import pandas as pd
import numpy as np
import random
import argparse 
import datetime
import json

from FormulaRetrieval import FormulaRetrieval
from EquationData import Equation

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            # print("h1_target:", h1_target.size())
            _, h2_target = self.get_target_encoder()(x2, edge_index2, edge_weight2)
            g1_target = global_add_pool(h1_target, batch)
            # print("g1_target:", g1_target.size())
            g2_target = global_add_pool(h2_target, batch)

        return g1, g2, h1_pred, h2_pred, g1_target, g2_target


def batch_detatch(batch_index, embs, emb_dict):
    tmp_dict = dict(zip(batch_index, embs))
    emb_dict.update(tmp_dict)

def get_embedding(encoder_model, dataloader):
    encoder_model.eval()
    emb = {}
    with torch.no_grad():    
        for data in dataloader:
            data = data.to(device_name)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            g1, g2, _, _, _, _ = encoder_model(data.x, data.edge_index, batch=data.batch)
            z = torch.cat([g1, g2], dim=1)
            
            batch_detatch(data.y, z.detach().cpu().numpy(), emb)
    # x = torch.cat(x, dim=0)

    return emb

def sum_collection(tensor_values_slt, tensor_values_opt):
    result = {}
    print(len(tensor_values_opt))
    print(len(tensor_values_slt))
    avg_opt = sum(tensor_values_opt.values())/float(len(tensor_values_opt))
    avg_slt = sum(tensor_values_slt.values())/float(len(tensor_values_slt))
    for formula_id in tensor_values_opt:
        temp = tensor_values_opt[formula_id]*0.2 + tensor_values_slt[formula_id]*0.8
        result[formula_id] = temp
    return result

def get_dataset(encode, pretrained, batch_size):
    judge_dataset = Equation(encode=encode, training=False, pretrained=pretrained, judge=True)
    judge_dataloader = DataLoader(judge_dataset, batch_size=batch_size, shuffle=True)
    query_dataset = Equation(encode=encode, pretrained=pretrained, training=False)
    query_dataloader = DataLoader(query_dataset, batch_size=20) 

    return judge_dataloader, query_dataloader


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=2048)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--pretrained", default=False, action='store_true')

    args = vars(parser.parse_args())

    batch_size = args['bs']
    run_id = args['run_id']
    pretrained = args['pretrained']

    if pretrained:
        input_dim = 200
        hidden_dim = 256
    else:
        input_dim = 2
        hidden_dim = 32

    slt_path = 'Retrieval_result/BGRL/slt/'+str(batch_size)+'/'+run_id+'/model'
    opt_path = 'Retrieval_result/BGRL/opt/'+str(batch_size)+'/'+run_id+'/model'

    device = torch.device(device_name)

    judge_data_opt, query_data_opt = get_dataset(encode='opt', pretrained=pretrained, batch_size=batch_size)
    judge_data_slt, query_data_slt = get_dataset(encode='slt', pretrained=pretrained, batch_size=batch_size)
        
    slt_path = "Retrieval_result/BGRL/slt/"+str(batch_size)+"/"+run_id+"/model"
    opt_path = "Retrieval_result/BGRL/opt/"+str(batch_size)+"/"+run_id+"/model"
    aug1 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.1)])
    aug2 = A.Compose([A.EdgeRemoving(pe=0.2), A.FeatureMasking(pf=0.1)])
    gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2).to(device)

    model_slt = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=256).to(device)
    model_opt = Encoder(encoder=gconv, augmentor=(aug1, aug2), hidden_dim=256).to(device)


    model_slt.load_state_dict(torch.load(slt_path), strict=False)
    model_opt.load_state_dict(torch.load(opt_path), strict=False)

    query_emb_slt = get_embedding(model_slt, query_data_slt)
    emb_dict_slt = get_embedding(model_slt, judge_data_slt)
    query_emb_opt = get_embedding(model_opt, query_data_opt)
    emb_dict_opt = get_embedding(model_opt, judge_data_opt)

    query_emb = sum_collection(query_emb_slt, query_emb_opt)
    emb_dict = sum_collection(emb_dict_slt, emb_dict_opt)

    result = FormulaRetrieval(emb_dict, query_emb, 1000)
    result.create_retrieval_file('combine', 'BGRL', batch_size, epoch='end', run_id=run_id)
   
    
if __name__== '__main__':
    main()


