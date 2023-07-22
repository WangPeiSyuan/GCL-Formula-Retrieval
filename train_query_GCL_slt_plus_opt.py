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
import datetime
import json


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
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        self.encoder = encoder

    def forward(self, x, edge_index, batch):
        z, g = self.encoder(x, edge_index, batch)
        return z, g

def batch_detatch(batch_index, embs, emb_dict):
    tmp_dict = dict(zip(batch_index, embs))
    emb_dict.update(tmp_dict)

def get_embedding(encoder_model, dataloader):
    emb = {}
    encoder_model.eval()
    with torch.no_grad():    
        for data in dataloader:
            data = data.to(device_name)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            _, g = encoder_model(data.x, data.edge_index, data.batch)
            batch_detatch(data.y, g.detach().cpu().numpy(), emb)

    return emb

def sum_collection(tensor_values_slt, tensor_values_opt):
    result = {}
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
        hidden_dim = 128
    else:
        input_dim = 2
        hidden_dim = 32

    slt_path = 'Retrieval_result/GCL/slt/'+str(batch_size)+'/'+run_id+'/model'
    opt_path = 'Retrieval_result/GCL/opt/'+str(batch_size)+'/'+run_id+'/model'

    device = torch.device(device_name)

    gconv = GConv(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2).to(device)
    model_slt = Encoder(encoder=gconv).to(device)
    model_opt = Encoder(encoder=gconv).to(device)
    model_slt.load_state_dict(torch.load(slt_path))
    model_opt.load_state_dict(torch.load(opt_path))
  
    judge_data_opt, query_data_opt = get_dataset(encode='opt', pretrained=pretrained, batch_size=batch_size)
    judge_data_slt, query_data_slt = get_dataset(encode='slt', pretrained=pretrained, batch_size=batch_size)
  

    query_emb_slt = get_embedding(model_slt, query_data_slt)
    emb_dict_slt = get_embedding(model_slt, judge_data_slt)
    query_emb_opt = get_embedding(model_opt, query_data_opt)
    emb_dict_opt = get_embedding(model_opt, judge_data_opt)
    
    query_emb = sum_collection(query_emb_slt, query_emb_opt)
    emb_dict = sum_collection(emb_dict_slt, emb_dict_opt)
    
    result = FormulaRetrieval(emb_dict, query_emb, 1000)
    result.create_retrieval_file('combine', 'GCL', batch_size, epoch='end', run_id=run_id)

if __name__== '__main__':
    main()

