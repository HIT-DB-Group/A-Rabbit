import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchbnn as bnn

from ML.AE.model.AutoEncoder import AutoEncoder

HIDDEN_DIMS=[[128,128], [128,128,64], [128,64,32], [128,64], [256,128], [256,128,64],[64],[128],[256]]
FINAL_DIMS=[[],[1],[0.5],[1,0.5],[1,1],[1,0.5,0.25]]
activation_function = [nn.Tanh(), nn.ReLU(), nn.LeakyReLU(),nn.Sigmoid()]

class Regression(nn.Module):
    def __init__(self,config):
        super(Regression,self).__init__()
        self.config=config
        self.encoder_model=AutoEncoder(config)
        pooling_dict={'mean':self._mean_pooling,'max':self._max_pooling,'attention':self._attention,'GNN':self._GNN,'root':self._root_node}
        self.pooling=pooling_dict[config['pooling']]
        dropout=0.1 if 'dropout' not in config else config['dropout']

        autoencoder_ffn=0 if 'autoencoder_ffn' not in config else config['autoencoder_ffn']
        AUTOENCODER_FFN=[[512,256,128,64],[256,128,64],[256,128,64,32],[512,256,128]]
        autoencoder_ffn=AUTOENCODER_FFN[autoencoder_ffn]
        index_scale=1 if 'index_scale' not in config else config['index_scale']
        index_autoencoder_ffn=[ int(i*index_scale) for i in autoencoder_ffn]

        input_dims=autoencoder_ffn[-1]+index_autoencoder_ffn[-1]
        act_fun=activation_function[0] if 'activation_function' not in config else activation_function[config['activation_function']]
        hidden_dims = [128,128] if 'hidden_dims' not in config else HIDDEN_DIMS[config['hidden_dims']]
        final_dims=[] if 'final_dims' not in config else FINAL_DIMS[config['final_dims']]
        self.encoders=nn.ModuleList()
        self.decoders=nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i==0:
                self.encoders.append(nn.Linear(input_dims, hidden_dims[i]))
            else:
                self.encoders.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            self.encoders.append(act_fun)
            self.encoders.append(nn.Dropout(dropout))

            self.decoders.insert(0, nn.Dropout(dropout))
            self.decoders.insert(0, act_fun)
            if i==0:
                self.decoders.insert(0, nn.Linear(hidden_dims[i], input_dims))
            else:
                self.decoders.insert(0, nn.Linear(hidden_dims[i], hidden_dims[i-1]))


        encoder_final_dim=hidden_dims[-1]
        final_dim=encoder_final_dim
        self.preds=nn.ModuleList()
        for i in range(len(final_dims)):
            if i==0:
                self.preds.append(nn.Linear(encoder_final_dim, int(encoder_final_dim*final_dims[i])))
            else:
                self.preds.append(nn.Linear(int(encoder_final_dim*final_dims[i-1]), int(encoder_final_dim*final_dims[i])))
            final_dim=int(encoder_final_dim*final_dims[i])
            self.preds.append(act_fun)
            self.preds.append(nn.Dropout(dropout))

        self.preds.append(nn.Linear(final_dim, 1))
        self.preds.append(act_fun)

    def forward(self, batched_data):
        original_feature,node_feature,origin_index_feature,index_feature,latent_feature,latent_index_feature=self.encoder_model.get_encoder(batched_data)

        final=self.pooling(latent_feature)
        index_final=self.pooling(latent_index_feature)
        final=torch.cat([final,index_final],dim=1)
        origin_index_query=final.clone()
        for encoder in self.encoders:
            final=encoder(final)
        index_query=final.clone()
        for decoder in self.decoders:
            index_query=decoder(index_query)
        for pred in self.preds:
            final=pred(final)
        
        uncertainty_dict={
            'query':(original_feature,node_feature),
            'index_query':(origin_index_query,index_query),
            'index_attention':(origin_index_feature,index_feature)
        }
        result_dict={
            'output':final,
            'uncertainty':uncertainty_dict
        }
        return result_dict
    
    def forward_uncertainty(self,batched_data):
        original_feature,node_feature,origin_index_feature,index_feature,latent_feature,latent_index_feature=self.encoder_model.get_encoder(batched_data)
        
        final=self.pooling(latent_feature)
        index_final=self.pooling(latent_index_feature)
        
        final=torch.cat([final,index_final],dim=1)
        origin_index_query=final.clone()
        for encoder in self.encoders:
            final=encoder(final)
        index_query=final.clone()
        for decoder in self.decoders:
            index_query=decoder(index_query)

        uncertainty_dict={
            'query':(original_feature,node_feature),
            'index_query':(origin_index_query,index_query),
            'index_attention':(origin_index_feature,index_feature)
        }
        return {'uncertainty':uncertainty_dict}

    def lock_encoder(self):
        self.encoder_model.lock_encoder()
        for param in self.encoders.parameters():
            param.requires_grad = False

    def _mean_pooling(self,plan_feature):    
        return torch.mean(plan_feature,dim=1)
    
    def _max_pooling(self,plan_feature):
        return torch.max(plan_feature,dim=1).values
    
    def _attention(self,plan_feature):
        pass

    def _GNN(self,plan_feature):
        pass

    def _root_node(self,plan_feature):
        return plan_feature[:,0,:].squeeze(1)

if __name__=='__main__':
    config = {
        'lr': 4.963736785618342e-05,
        'bs': 512,
        'epochs': 150,
        'clip_size': 32,
        'emb_size': 16,
        'pred_hid': 128,
        'ffn_dim': 256,
        'head_size': 4,
        'n_layers': 16,
        'join_schema_head_size': 16,
        'attention_dropout_rate': 0.024913959488555452,
        'join_schema_layers': 3,
        'dropout': 0.24206314591185082,
        'sch_decay': 0.6261264298150304,
        'predict_contract': 1,
        'midlayers': False,
        'res_con': True,
        'encoder_mode': 'onehot',
        'pooling': 'max',
        'position_encoding': 'position_encoder',
        'device': 'cpu',
        'hidden_dims': 1,
        'final_dims': 4,
        'activation_function': 0,
        'autoencoder_ffn': 3,
        'index_scale': 0.5,
        'use_weight_mseloss': True,
        'log_process': True,
        'is_tpcds': True
    }
    regression=Regression(config)
    print(regression)
