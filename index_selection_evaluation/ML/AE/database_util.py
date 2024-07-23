import numpy as np
import pandas as pd
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
import time


class Batch():
    def __init__(self, features,pad_mask, attention_bias,attention_bias_SuperNode, join_schema_bias,join_features,heights,index_features,index_pad_mask):
        super(Batch, self).__init__()

        self.features = features
        self.pad_mask=pad_mask
        self.attention_bias = attention_bias
        self.attention_bias_SuperNode=attention_bias_SuperNode
        self.join_schema_bias = join_schema_bias
        self.heights=heights
        self.join_features=join_features
        self.index_features=index_features
        self.index_pad_mask=index_pad_mask
        
    def to(self, device):

        self.features = self.features.to(device)
        self.pad_mask=self.pad_mask.to(device)
        self.attention_bias = self.attention_bias.to(device)
        self.attention_bias_SuperNode = self.attention_bias_SuperNode.to(device)        
        self.join_schema_bias = self.join_schema_bias.to(device)
        self.heights=self.heights.to(device)
        self.join_features=self.join_features.to(device)
        self.index_features=self.index_features.to(device)
        self.index_pad_mask=self.index_pad_mask.to(device)

        return self

    def __len__(self):
        return self.features.shape[0]
    
    def replace_index_features(self,key, index_features):
        
        
        index_features=torch.tensor(index_features, dtype=torch.float)
        index_features=index_features.unsqueeze(0).repeat(len(key),1,1)
        index_pad_mask=torch.ones([len(key),index_features.shape[1]], dtype=torch.float)
        
        return self.features[key],self.pad_mask[key],self.attention_bias[key],self.attention_bias_SuperNode,self.join_schema_bias[key],self.join_features,self.heights[key],index_features,index_pad_mask
        

def collator(small_set,log_process=False):
    y_ratios = small_set[1]
    y_reals=small_set[2]
    if log_process:
        y_ratios=torch.tensor(y_ratios, dtype=torch.float)
        y_ratios=torch.log(2-y_ratios)
    else:
        y_ratios=torch.tensor(y_ratios, dtype=torch.float)


    length=np.array([s['length'] for s in small_set[0]])
    maxlength=np.max(length)
    pad_length=maxlength-length

    features=[np.pad(s['features'], ((0, pad_length[i]), (0, 0)), mode='constant', constant_values=-1) for i,s in enumerate(small_set[0])]
    features=torch.tensor(np.stack(features), dtype=torch.float)+1
    features[:,:,[-1,-2]]=features[:,:,[-1,-2]]-1
    features[features<0]=0

    pad_mask=[np.pad(np.ones(s['length']), ((0, pad_length[i])), mode='constant', constant_values=0) for i,s in enumerate(small_set[0])]
    pad_mask=torch.tensor(np.stack(pad_mask), dtype=torch.float)
    

    attention_bias=[np.pad(s['attention_bias'], ((0, pad_length[i]), (0, pad_length[i])), mode='constant', constant_values=-1) for i,s in enumerate(small_set[0])]
    attention_bias=torch.tensor(np.stack(attention_bias), dtype=torch.float)+1

    attention_bias_SuperNode=torch.zeros([maxlength+1,maxlength+1], dtype=torch.float)
    
    join_schema_bias=[np.pad(s['join_schema_bias'], ((1,0), (1,0)), mode='constant', constant_values=-1) for i,s in enumerate(small_set[0])]
    join_schema_bias=torch.tensor(np.stack(join_schema_bias), dtype=torch.float)+1
    join_features=torch.tensor([i for i in range(join_schema_bias.shape[1])], dtype=torch.float)
    

    index_length=np.array([s['index_length'] for s in small_set[0]])
    max_index_length=np.max(index_length)
    index_pad_length=max_index_length-index_length


    index_features=[np.pad(s['indexes_features'], ((0, index_pad_length[i]), (0, 0)), mode='constant', constant_values=0) for i,s in enumerate(small_set[0])]

    index_features=torch.tensor(np.stack(index_features), dtype=torch.float)

    index_pad_mask=[np.pad(np.ones(s['index_length']), ((0, index_pad_length[i])), mode='constant', constant_values=0) for i,s in enumerate(small_set[0])]
    index_pad_mask=torch.tensor(np.stack(index_pad_mask), dtype=torch.float)

    heights=[np.pad(s['heights'], ((0, pad_length[i])), mode='constant', constant_values=-1) for i,s in enumerate(small_set[0])]
    heights=torch.tensor(np.stack(heights), dtype=torch.float)+1
    bth=Batch(features,pad_mask, attention_bias,attention_bias_SuperNode, join_schema_bias,join_features,heights,index_features,index_pad_mask)
    batch_dict={
        'batch':bth,
        'y_ratios':y_ratios,
        'y_reals':torch.tensor(y_reals, dtype=torch.float),
        'length':length,
        'index_length':index_length,
    }
    return batch_dict
