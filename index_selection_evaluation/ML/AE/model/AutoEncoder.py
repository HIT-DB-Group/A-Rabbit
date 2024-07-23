import torch
from torch.utils.data import Dataset
import json
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

def tensor_size(tensor):
    return tensor.element_size() * tensor.nelement()

def get_tensor_size_in_MB(tensor):
    byte_size = tensor_size(tensor)
    return byte_size / (1024 ** 2)

class AddPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_length=512):
        super(AddPositionalEncoding, self).__init__()
        self.encoding = self.generate_positional_encoding(embedding_dim, max_seq_length)
        self.encoding.requires_grad = False
        self.encoding=self.encoding.to('cuda')
    def forward(self, x):
        seq_length = x.size(1)


        self.encoding[:, :seq_length].detach().to(x.device)
        x = x + self.encoding[:, :seq_length].detach().to(x.device)
        return x

    def generate_positional_encoding(self, embedding_dim, max_seq_length=512):
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        pos_encoding = torch.zeros((max_seq_length, embedding_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        return pos_encoding



class DropoutMC(torch.nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.dropout(
            x, self.p, training=self.training or self.activate
        )

AUTOENCODER_FFN=[[512,256,128,64],[256,128,64],[256,128,64,32],[512,256,128]]

class AutoEncoder(nn.Module):
    def __init__(self, config):
        
        super(AutoEncoder,self).__init__()
        emb_size=32 if 'emb_size' not in config else config['emb_size']
        ffn_dim=32 if 'ffn_dim' not in config else config['ffn_dim']
        head_size=8 if 'head_size' not in config else config['head_size']
        join_schema_head_size=2 if 'join_schema_head_size' not in config else config['join_schema_head_size']
        dropout=0.1 if 'dropout' not in config else config['dropout']
        attention_dropout_rate=0.1 if 'attention_dropout_rate' not in config else config['attention_dropout_rate']
        n_layers=8 if 'n_layers' not in config else config['n_layers']
        join_schema_layers=2 if 'join_schema_layers' not in config else config['join_schema_layers']
        pred_hid=256 if 'pred_hid' not in config else config['pred_hid']
        predict_contract=1 if 'predict_contract' not in config else config['predict_contract']
        midlayers=True if 'midlayers' not in config else config['midlayers']
        res_con=True if 'res_con' not in config else config['res_con']
        self.position_encoding='height_encoder' if 'position_encoding' not in config else config['position_encoding']
        index_scale=1 if 'index_scale' not in config else config['index_scale']
        autoencoder_ffn=0 if 'autoencoder_ffn' not in config else config['autoencoder_ffn']
        encoder_mode=config['encoder_mode']

        index_emb_size=int(emb_size*index_scale)
        index_ffn_dim=int(ffn_dim*index_scale)
        index_head_size=max(int(head_size*index_scale),1)

        autoencoder_ffn=AUTOENCODER_FFN[autoencoder_ffn]
        index_autoencoder_ffn=[ int(i*index_scale) for i in autoencoder_ffn]

        relu_in_decoder=True if 'relu_in_decoder' not in config else config['relu_in_decoder']

        is_tpcds=False if 'is_tpcds' not in config else config['is_tpcds']
        
        
        if encoder_mode=='embedding':
            self.feature_encoder=FeatureEmbed_nolinear(emb_size)
            self.index_encoder=IndexesEmbed_nolinear(index_emb_size,position_num=2)
        elif encoder_mode=='onehot':
            if is_tpcds:
                self.feature_encoder=OnehotFeatureEmbed(column_num=300)
                self.index_encoder=IndexOnehotFeatureEmbed(column_num=300,position_num=2)
            else:
                self.feature_encoder=OnehotFeatureEmbed()
                self.index_encoder=IndexOnehotFeatureEmbed(position_num=2)
        else:
            raise Exception(f'encoder_mode:{encoder_mode}不合法')
        
        self.embed_sizes=[self.feature_encoder.get_output_size()]
        self.embed_sizes.extend(autoencoder_ffn)
        self.position_encoder=AddPositionalEncoding(self.feature_encoder.get_output_size())
        self.height_encoder = nn.Embedding(64, self.feature_encoder.get_output_size(), padding_idx=0)
        encoders=[]
        decoders=[]

        for i in range(len(self.embed_sizes)-1):
            encoders.append(EncoderLayer(self.embed_sizes[i], ffn_dim, dropout, attention_dropout_rate, head_size))
            encoders.append(nn.Linear(self.embed_sizes[i],self.embed_sizes[i+1]))
            encoders.append(nn.ReLU())
            decoders.insert(0,EncoderLayer(self.embed_sizes[i], ffn_dim, dropout, attention_dropout_rate, head_size))
            decoders.insert(0,nn.ReLU())
            decoders.insert(0,nn.Linear(self.embed_sizes[i+1],self.embed_sizes[i]))
        
        decoders.append(nn.Linear(self.feature_encoder.get_output_size(),12))

        if relu_in_decoder:
            decoders.append(nn.ReLU())

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        

        self.index_embed_sizes=[self.index_encoder.get_output_size()]
        self.index_embed_sizes.extend(index_autoencoder_ffn)
        index_encoders=[]
        index_decoders=[]
        for i in range(len(self.index_embed_sizes)-1):
            index_encoders.append(EncoderLayer(self.index_embed_sizes[i], index_ffn_dim, dropout, attention_dropout_rate, index_head_size))
            index_encoders.append(nn.Linear(self.index_embed_sizes[i],self.index_embed_sizes[i+1]))
            index_encoders.append(nn.ReLU())
            index_decoders.insert(0,EncoderLayer(self.index_embed_sizes[i], index_ffn_dim, dropout, attention_dropout_rate, index_head_size))
            index_decoders.insert(0,nn.ReLU())
            index_decoders.insert(0,nn.Linear(self.index_embed_sizes[i+1],self.index_embed_sizes[i]))
        
        index_decoders.append(nn.Linear(self.index_encoder.get_output_size(),2))
        if relu_in_decoder:
            index_decoders.append(nn.ReLU())

        self.index_encoders = nn.ModuleList(index_encoders)
        self.index_decoders = nn.ModuleList(index_decoders)


        self.tree_bias_encoder = nn.Embedding(64, head_size, padding_idx=0)

    def lock_encoder(self):
        for param in self.encoders.parameters():
            param.requires_grad = False
        for param in self.index_encoders.parameters():
            param.requires_grad = False
        for param in self.feature_encoder.parameters():
            param.requires_grad = False
        for param in self.position_encoder.parameters():
            param.requires_grad = False
        for param in self.height_encoder.parameters():
            param.requires_grad = False
        for param in self.tree_bias_encoder.parameters():
            param.requires_grad = False



    def get_encoder(self,batched_data):
        features=batched_data.features
        index_features=batched_data.index_features
        pad_mask=batched_data.pad_mask
        index_pad_mask=batched_data.index_pad_mask
        attention_bias = batched_data.attention_bias
        heights = batched_data.heights
        n_batch=features.shape[0]
        original_feature=features.clone()
        origin_index_feature=index_features.clone()
        attention_bias=self.tree_bias_encoder(attention_bias.long()).permute(0, 3, 1, 2)
        node_feature = self.feature_encoder(features)
        
        
        if self.position_encoder=='height_encoder':
            node_feature = node_feature + self.height_encoder(heights.long())
        else:
            node_feature = self.position_encoder(node_feature)

        pad_mask=pad_mask.unsqueeze(-1)
        pad_mask=pad_mask.repeat(1,1,self.feature_encoder.get_output_size())
        assert node_feature.shape==pad_mask.shape,print(f'node_feature.shape:{node_feature.shape},pad_mask.shape:{pad_mask.shape}')
        node_feature=node_feature*pad_mask
        
        for enc_layer in self.encoders:
            if isinstance(enc_layer,EncoderLayer):
                node_feature = enc_layer(node_feature, attention_bias)
            else:
                node_feature=enc_layer(node_feature)

        latent_feature=node_feature.clone()

        for dec_layer in self.decoders:
            if isinstance(dec_layer,EncoderLayer):
                node_feature = dec_layer(node_feature, attention_bias)
            else:
                node_feature=dec_layer(node_feature)
        

        index_feature=self.index_encoder(index_features)

        index_pad_mask=index_pad_mask.unsqueeze(-1)
        index_pad_mask=index_pad_mask.repeat(1,1,self.index_encoder.get_output_size())
        assert index_feature.shape==index_pad_mask.shape,print(f'index_feature.shape:{index_feature.shape},index_pad_mask.shape:{index_pad_mask.shape}')
        index_feature=index_feature*index_pad_mask

        for enc_layer in self.index_encoders:
            index_feature=enc_layer(index_feature)

        latent_index_feature=index_feature.clone()

        for dec_layer in self.index_decoders:
            index_feature=dec_layer(index_feature)
        

        return original_feature,node_feature,origin_index_feature,index_feature,latent_feature,latent_index_feature

class Prediction(nn.Module):
    def __init__(self, in_feature = 69, hid_units = 256, contract = 1, mid_layers = True, res_con = False,dropout=0.01):
        super(Prediction, self).__init__()
        self.mid_layers = mid_layers
        self.res_con = res_con
        
        self.out_mlp1 = nn.Linear(in_feature, hid_units)
        self.mid_mlp1 = nn.Linear(hid_units, int(hid_units//contract))
        self.mid_mlp2 = nn.Linear(int(hid_units//contract), hid_units)
        self.mid_mlp3 = nn.Linear(hid_units, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)
        self.mcdropout = DropoutMC(dropout,activate=True)

    def forward(self, features):
        
        hid = F.relu(self.mcdropout(self.out_mlp1(features)))
        if self.mid_layers:
            mid = F.relu(self.mcdropout(self.mid_mlp1(hid)))
            mid = F.relu(self.mcdropout(self.mid_mlp2(mid)))
            mid = F.relu(self.mcdropout(self.mid_mlp3(mid)))
            if self.res_con:
                hid = hid + mid
            else:
                hid = mid

        out = self.out_mlp2(hid)

        return out

class IndexOnehotFeatureEmbed(nn.Module):
    def __init__(self,column_num=128,position_num=2):
        super(IndexOnehotFeatureEmbed, self).__init__()
        self.column_num=column_num
        self.position_num=position_num

    def forward(self, index_feature):
        self.device=index_feature.device
        index_emb = self.getColumn(index_feature.long())
        final=index_emb.reshape(index_emb.shape[0],index_emb.shape[1],-1)
        final=final.float()
        return final
    
    def getColumn(self, columnId):
        emb = F.one_hot(columnId,num_classes=self.column_num)
        return emb
    
    def get_output_size(self):
        size =self.column_num*self.position_num
        return size

class OnehotFeatureEmbed(nn.Module):
    def __init__(self, type_num=32, table_num=27,column_num=128):
        super(OnehotFeatureEmbed, self).__init__()

        self.type_num=type_num
        self.table_num=table_num
        self.column_num=column_num
        self.weight=self._calculate_weight()

    def forward(self, feature):
        self.device=feature.device
        typeId,tableId,columnsIds,joinId,joinColumnsIds,cost_card=torch.split(feature,(1,1,2,3,3,2), dim = -1)
        typeEmb = self.getType(typeId.long())        
        bt,sq,ta=joinId.shape
        tableEmb=self.getTable(tableId.long())
        tableEmb=tableEmb.reshape(bt,sq,-1)

        columnsEmb = self.getColumn(columnsIds.long())
        columnsEmb=columnsEmb.reshape(bt,sq,-1)

        joinEmb=self.getTable(joinId.long())
        joinEmb=joinEmb.reshape(bt,sq,-1)

        joinColumnsEmb = self.getColumn(joinColumnsIds.long())
        joinColumnsEmb=joinColumnsEmb.reshape(bt,sq,-1)

        final = torch.cat((typeEmb, tableEmb, columnsEmb, joinEmb, joinColumnsEmb, cost_card), dim = 2)
        final=final.float()
        return final
    
    def getType(self, typeId):
        emb = F.one_hot(typeId,num_classes=self.type_num)
        return emb.squeeze(2)

    def getColumn(self, columnId):
        emb = F.one_hot(columnId,num_classes=self.column_num)
        return emb
    
    def getTable(self,tableId):
        emb=F.one_hot(tableId,num_classes=self.table_num)
        return emb
    
    def get_output_size(self):
        size = self.type_num + self.table_num + self.column_num*2 + self.table_num*3 + self.column_num*3 + 2
        return size
    def _calculate_weight(self):
        weight=torch.ones(self.type_num)/self.type_num
        weight=torch.cat((weight,torch.ones(self.table_num)/self.table_num),dim=0)
        weight=torch.cat((weight,torch.ones(self.column_num*2)/self.column_num),dim=0)
        weight=torch.cat((weight,torch.ones(self.table_num*3)/self.table_num),dim=0)
        weight=torch.cat((weight,torch.ones(self.column_num*3)/self.column_num),dim=0)
        weight=torch.cat((weight,torch.ones(2)),dim=0)
        return weight**0.5

    def get_weight(self):
        return self.weight.to(self.device)

class FeatureEmbed_nolinear(nn.Module):
    def __init__(self, embed_size=32, types=32,table_num=27,columns=300):
        super(FeatureEmbed_nolinear, self).__init__()
        self.embed_size=embed_size
        self.typeEmbed = nn.Embedding(types, embed_size,padding_idx=0)
        self.tableEmbed = nn.Embedding(table_num, embed_size,padding_idx=0)
        self.columnEmbed = nn.Embedding(columns, embed_size,padding_idx=0)

    def forward(self, feature):
        typeId,tableId,columnsIds,joinId,joinColumnsIds,cost_card=torch.split(feature,(1,1,2,3,3,2), dim = -1)
        typeEmb = self.getType(typeId)        
        
        bt,sq,ta=joinId.shape
        joinId=joinId.reshape(bt,-1)

        joinEmb=self.getTable(joinId)
        joinEmb=joinEmb.reshape(bt,sq,-1)

        tableEmb=self.getTable(tableId)
        tableEmb=tableEmb.view(bt,sq,-1)
        
        columnsEmb = self.getColumn(columnsIds)
        columnsEmb=columnsEmb.reshape(bt,sq,-1)
        joinColumnsEmb = self.getColumn(joinColumnsIds)
        joinColumnsEmb=joinColumnsEmb.reshape(bt,sq,-1)

        final = torch.cat((typeEmb, tableEmb, columnsEmb, joinEmb, joinColumnsEmb, cost_card), dim = 2)
        
        return final
    def getTable(self,tableId):
        emb = self.tableEmbed(tableId.long())
        return emb

    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())
        return emb.squeeze(2)

    def getColumn(self, columnId):
        emb = self.columnEmbed(columnId.long())
        return emb
    
    def get_output_size(self):
        size = self.embed_size*10 +1 +1
        return size

class FeatureEmbed(nn.Module):
    def __init__(self, embed_size=32, types=32,columns=300):
        super(FeatureEmbed, self).__init__()

        self.embed_size=embed_size
        self.typeEmbed = nn.Embedding(types, embed_size,padding_idx=0)
        self.columnEmbed = nn.Embedding(columns, embed_size,padding_idx=0)
        

        self.project = nn.Linear(embed_size*10 +1 +1 , embed_size*10 +1 +1 )

    def forward(self, feature, join_tables):
        typeId,tableId,columnsIds,joinId,joinColumnsIds,cost_card=torch.split(feature,(1,1,2,3,3,2), dim = -1)
        typeEmb = self.getType(typeId.long())        
        
        bt,sq,ta=joinId.shape
        joinId=joinId.reshape(bt,-1).long()

        joinEmb=join_tables[torch.arange(join_tables.size(0)).unsqueeze(1), joinId].squeeze(-2)
        joinEmb=joinEmb.reshape(bt,sq,-1)

        bt,sq,ta=tableId.shape
        tableEmb=join_tables[torch.arange(join_tables.size(0)).unsqueeze(1), tableId.view(bt,-1).long()].squeeze(-2)
        tableEmb=tableEmb.view(bt,sq,-1)
        
        columnsEmb = self.getColumn(columnsIds)
        columnsEmb=columnsEmb.reshape(bt,sq,-1)
        joinColumnsEmb = self.getColumn(joinColumnsIds)
        joinColumnsEmb=joinColumnsEmb.reshape(bt,sq,-1)


        final = torch.cat((typeEmb, tableEmb, columnsEmb, joinEmb, joinColumnsEmb, cost_card), dim = 2)
        final = F.leaky_relu(self.project(final))
        
        return final
    
    def getType(self, typeId):
        emb = self.typeEmbed(typeId.long())
        return emb.squeeze(2)

    def getColumn(self, columnId):
        emb = self.columnEmbed(columnId.long())
        return emb
    
    def get_output_size(self):
        size = self.embed_size*10 +1 +1
        return size


class TableEmbed(nn.Module):
    def __init__(self, embed_size=32,output_dim=32,n_tables=32):
        super(TableEmbed, self).__init__()
        self.tableEmbed = nn.Embedding(n_tables, embed_size)
        self.linearTable = nn.Linear(embed_size, output_dim)
        
    
    def forward(self, feature):
        output=self.tableEmbed(feature)
        tables_embedding=F.leaky_relu(self.linearTable(output))
        return tables_embedding

class IndexesEmbed_nolinear(nn.Module):
    def __init__(self, embed_size=32, columns=300, position_num=40,hid_dim=256):
        super(IndexesEmbed_nolinear, self).__init__()

        self.embed_size=embed_size
        self.position_num=position_num
        self.indexEmb = nn.Embedding(columns, embed_size,padding_idx=0)

    def forward(self, feature):
        final = self.getIndex(feature)
        final = final.reshape(final.shape[0],final.shape[1],-1)
        return final

    def getIndex(self, indexID):
        emb = self.indexEmb(indexID.long())
        return emb
    
    
    def get_output_size(self):
        size = self.embed_size*self.position_num
        return size

class IndexesEmbed(nn.Module):
    def __init__(self, embed_size=32, columns=300, position_num=40,hid_dim=256):
        super(IndexesEmbed, self).__init__()

        self.embed_size=embed_size
        self.position_num=position_num
        self.indexEmb = nn.Embedding(columns, embed_size,padding_idx=0)

        self.project = nn.Linear(embed_size*position_num , embed_size*position_num )
        self.mid1=nn.Linear(embed_size*position_num,hid_dim)
        self.output=nn.Linear(hid_dim,embed_size)

    def forward(self, feature):
        final = self.getIndex(feature)

        final = final.reshape(final.shape[0],-1)
        final = F.leaky_relu(self.project(final))
        final = F.leaky_relu(self.mid1(final))
        final = F.leaky_relu(self.output(final))
        
        return final

    def getIndex(self, indexID):
        emb = self.indexEmb(indexID.long())
        return emb
    
    
    def get_output_size(self):
        size = self.embed_size
        return size



class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()
        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5


        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = DropoutMC(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):

        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = DropoutMC(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = DropoutMC(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x






















