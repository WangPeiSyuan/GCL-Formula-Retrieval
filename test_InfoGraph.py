import torch
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from torch.optim import Adam
from GCL.models import SingleBranchContrast
from torch_geometric.nn import  GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import uniform

from copy import deepcopy
import pandas as pd
import numpy as np
import random
import argparse 
import datetime
import json

from FormulaRetrieval import FormulaRetrieval
from EquationData import Equation

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

def get_embedding(encoder_model, dataloader):
    emb = {}
    encoder_model.eval()
    for data in dataloader:
        data = data.to('cuda')
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        _, g = encoder_model(data.x, data.edge_index, data.batch)
        batch_detatch(data.y, g.detach().cpu().numpy(), emb)
    return emb

def sum_collection(tensor_values_slt, tensor_values_opt):
    result = {}
    print(len(tensor_values_opt))
    print(len(tensor_values_slt))
    avg_opt = sum(tensor_values_opt.values())/float(len(tensor_values_opt))
    avg_slt = sum(tensor_values_slt.values())/float(len(tensor_values_slt))
    for formula_id in tensor_values_opt:
        temp = tensor_values_opt[formula_id]*0.9 + tensor_values_slt[formula_id]*0.1
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

    
    slt_path = 'Retrieval_result/InfoG/slt/'+str(batch_size)+'/'+run_id+'/model'
    opt_path = 'Retrieval_result/InfoG/opt/'+str(batch_size)+'/'+run_id+'/model'

    device = torch.device('cuda')

    gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, activation=torch.nn.ReLU, num_layers=2).to(device)
    fc1 = FC(hidden_dim=hidden_dim*2)
    fc2 = FC(hidden_dim=hidden_dim*2)

    model_slt = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)
    model_opt = Encoder(encoder=gconv, local_fc=fc1, global_fc=fc2).to(device)


    model_slt.load_state_dict(torch.load(slt_path), strict=False)
    model_opt.load_state_dict(torch.load(opt_path), strict=False)

    
    judge_data_opt, query_data_opt = get_dataset(encode='opt', pretrained = pretrained, batch_size=batch_size)
    judge_data_slt, query_data_slt = get_dataset(encode='slt', pretrained = pretrained, batch_size=batch_size)

    query_emb_slt = get_embedding(model_slt, query_data_slt)
    emb_dict_slt = get_embedding(model_slt, judge_data_slt)
    query_emb_opt = get_embedding(model_opt, query_data_opt)
    emb_dict_opt = get_embedding(model_opt, judge_data_opt)

    query_emb = sum_collection(query_emb_slt, query_emb_opt)
    emb_dict = sum_collection(emb_dict_slt, emb_dict_opt)

    result = FormulaRetrieval(emb_dict, query_emb, 1000)
    result.create_retrieval_file('combine', 'InfoG', batch_size=batch_size, epoch='end', run_id=run_id)


    


