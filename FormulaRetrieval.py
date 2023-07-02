import os
import torch
import os.path as osp
import torch.nn.functional as F
import numpy as np

class FormulaRetrieval(): 
    def __init__(self, series, query_emb, topk=100):
        super(FormulaRetrieval, self).__init__()
        self.retrieval_result = {}
        self.emb_dict_query = query_emb
        self.emb_dict_series = series
        self.retrieval(self.emb_dict_query, self.emb_dict_series, topk)
        # self.create_retrieval_file(run_id)
        
    def batch_detatch(self, batch_index, embs, emb_dict):
        tmp_dict = dict(zip(batch_index, embs))
        emb_dict.update(tmp_dict)
        return emb_dict
   
    def retrieval(self, emb_dict_query, emb_dict_series, k):
        print("retrieval...")
        formula_index = list(emb_dict_series.keys())
        series_tensor = torch.as_tensor(np.array(list(emb_dict_series.values())))        
        for query_key in emb_dict_query:
            query = emb_dict_query[query_key]
            query_tensor = torch.tensor(query).double()
            self.retrieval_result[query_key] = self.get_formula_retrieval(series_tensor, formula_index, query_tensor)
#             break
    def get_formula_retrieval(self, series_tensor, formula_index, query_tensor):
        dist = F.cosine_similarity(series_tensor, query_tensor)
        index_sorted = torch.sort(dist, descending=True)[1]
        top_1000 = index_sorted[:1000]
        top_1000 = top_1000.data.cpu().numpy()
        cos_values = torch.sort(dist, descending=True)[0][:1000].data.cpu().numpy()
        result = {}
        count = 0
        for x in top_1000:
            doc_id = formula_index[x]
            score = cos_values[count]
            result[doc_id] = score
            count += 1
        return result

    def create_retrieval_file(self, encode, model, batch_size, epoch, run_id):
        print("create result file...")
        filepath = "Retrieval_result/"+str(model)+"/"+encode+"/"+str(batch_size)+"/"+str(run_id)
        filename = "/retrieval_res"+str(run_id)+"_"+str(epoch)
        try:
            os.makedirs(filepath)
            print(filepath)
        except:
            pass
        file = open(filepath+filename, "w")
        for query in self.retrieval_result:
            # print(query)
            count = 1
            line = query + " xxx "
            for s in self.retrieval_result[query]:
                score = self.retrieval_result[query][s]
                temp = line + s  + " " + str(count) + " " + str(score) + " Run_" + str(run_id)
                # temp = line + s  + " " + str(count) + " " + str(score)
                count += 1
                file.write(temp + "\n")
        file.close()
    
    def cosine_similarity(self, A, B):
        return F.cosine_similarity(A, B)
    
                