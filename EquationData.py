import numpy as np
import torch
import json

from torch.utils.data import Dataset
from torch_geometric.data import Data

class Equation(Dataset):
    def __init__(self, encode='opt', training=True, judge=False, pretrained=False):
       
        self.pretrained = pretrained
        self.ids=[]
        self.graphs=[]

        
        file_path = 'datasets/encoder/'
        # get char embedding file
        if pretrained:
            filename = file_path + encode + '_char_embedding.txt'
            with open(filename) as f:
                self.char_emb = json.loads(f.read())
            
        filename = encode+'_list.txt'
        filename = file_path + filename
        with open(filename) as f:
            self.formula = json.loads(f.read())   
        self.edge_dict = self.process_dict()
        if not pretrained:
            self.type_dict, self.val_dict = self.process_node_dict()
        
        if training:
            self.process_sample(self.formula)
        else:
            if judge:
                filename = encode+'_judge.txt'
            else:
                filename = 'query_' + encode + '_list.txt'
            filename = file_path + filename
            with open(filename) as f:
                query = json.loads(f.read())
            self.process_sample(query)
       

    def process_sample(self, data):
        print("processing graph sample...")
        idx = 0
        for key, series_data in data.items():
            data = self._get_graph(key, series_data)
            self.ids.append(key)    
            self.graphs.append(data)   
            idx+=1
            
    def _get_graph(self, key, inter):
        src=[]
        dst=[]
        edges=[]
        node_dict={}
        for idx, rel in enumerate(inter):
            node1, node2, edge = rel
            src.append(node1[2])
            dst.append(node2[2])
            edges.append([self.edge_dict[edge[0]]])
            # edges.append(edge[0])
            if(node1[2] not in node_dict):
                if self.pretrained:
                    node_dict[node1[2]] = self.char_emb[str(node1[0])] + self.char_emb[str(node1[1])]
                else:
                    node_dict[node1[2]] = [self.type_dict[node1[0]],  self.val_dict[node1[1]]]
            if(node2[2] not in node_dict):
                if self.pretrained:
                    node_dict[node2[2]] = self.char_emb[str(node2[0])] + self.char_emb[str(node2[1])]
                else:
                    node_dict[node2[2]] = [self.type_dict[node2[0]], self.val_dict[node2[1]]]
        edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
        # node_dict = dict(sorted(node_dict.items()))
        attr = np.vstack((list(node_dict.values())))
        x = torch.as_tensor(attr, dtype=torch.float)
        edge_attr = torch.as_tensor(edges, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=key)
        return data
    
    def process_dict(self):
        edge_dict = {}
        for key in self.formula:
            for tp in self.formula[key]:
                node1, node2, edge = tp
                if(edge[0] not in edge_dict):
                    edge_dict[edge[0]] = 1
        edge_dict = {key: i for i, key in enumerate(edge_dict)}
        return edge_dict
    
    def process_node_dict(self):
        type_dict = {}
        val_dict = {}
        for key in self.formula:
            for tp in self.formula[key]:
                node1, node2, edge = tp 
                if(node1[0] not in type_dict):
                    type_dict[node1[0]] = 1
                if(node1[1] not in val_dict):
                    val_dict[node1[1]] = 1
                if(node2[0] not in type_dict):
                    type_dict[node2[0]] = 1
                if(node2[1] not in val_dict):
                    val_dict[node2[1]] = 1
        type_dict = {key: i for i, key in enumerate(type_dict)}
        val_dict = {key: i for i, key in enumerate(val_dict)}
        return type_dict, val_dict
                
                
        
    def __getitem__(self, i):
        return self.graphs[i]
     
    def __len__(self):
        return len(self.graphs)

    def avg_num_node(self):
        num_node = []
        for key in self.formula:
            src = []
            dst = []
            for rel in self.formula[key]:
                node1, node2, edge = rel
                src.append(node1[2])
                dst.append(node2[2])
            count = len(list(set(src+dst)))
            num_node.append(count)
            
        return sum(num_node)/len(num_node)
        
