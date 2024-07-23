import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import json
import pandas as pd
import sys, os, copy,logging
from collections import deque
import time
from scipy.stats import pearsonr
import pickle
import random
import re


from ML.AE.database_util import *
from selection import constants
from selection.utils import *
from selection.workload import ProcessedWorkload
from selection.index import Index


phy2log_map=constants.phy2log_map
indexed_type=constants.indexed_type
can_index_column_tpye=constants.can_index_column_tpye
can_index_operator_str=constants.can_index_operator_str

class Encoding:
    def __init__(self, pWorkload:ProcessedWorkload):
        self.pWorkload=pWorkload
        self.node_types2idx={'Unique':0,'Hash Join':1,'Bitmap Heap Scan':2,'Materialize':3,'SetOp':4,'Subquery Scan':5,'Aggregate':6,'BitmapAnd':7,'Gather Merge':8,'WindowAgg':9,'Sort':10,'Gather':11,'Index Scan':12,'Merge Join':13,'Bitmap Index Scan':14,'Nested Loop':15,'Index Only Scan':16,'CTE Scan':17,'Hash':18,'BitmapOr':19,'Limit':20,'Result':21,'Merge Append':22,'Append':23,'Group':24,'Seq Scan':25}

        tablesid2sort=[(degree+i*0.0001,i) for i,degree in enumerate(np.sum(self.pWorkload.join_schema,axis=0))]
        list.sort(tablesid2sort,key=lambda x:x[0],reverse=True)
        self.tablesid2idx={id:i for i,(_,id) in enumerate(tablesid2sort)}
        tablesExc=[ id for _,id in tablesid2sort]
        self.join_schema_bias=self.pWorkload.join_schema[tablesExc,:]
        self.join_schema_bias=self.join_schema_bias[:,tablesExc]

        self.tablesid2idx[None]=-1
        self.join_schema_bias=np.pad(self.join_schema_bias,((0,1),(0,1)),'constant',constant_values=0)

        self.none_column_idx=-1


        self.index_change2idx={(False,False):0,(True,False):1,(True,True):2,(False,True):3}

    def get_plan_encoder(self,plan,new_indexes):
        new_indexes=list(new_indexes)
        new_indexes_idx=[]
        for indexes in new_indexes:
            names=indexes._column_names()
            new_indexes_idx.append([self.pWorkload.indexable_columns_dict[name] for name in names])
        
        indexes_features=[]
        for indexes in new_indexes_idx:
            idf=[0,0]
            for i,idx in enumerate(indexes):
                idf[i]=idx+1
            indexes_features.append(idf)
        indexes_features=np.array(indexes_features,dtype=float)
        index_length=len(new_indexes_idx)
        if len(indexes_features)==0:
            indexes_features=np.array([[0,0]],dtype=float)
            index_length=1

        features=np.zeros((len(plan.dfs_nodes),12),dtype=float)
        for i,node in enumerate(plan.dfs_nodes):
            features[i][0]=self.node_types2idx[node.nodeType]
            features[i][1]=self.tablesid2idx[node.table]
            l=len(node.columns)
            for j in range(2):
                if j>=l:
                    features[i][2+j]=self.none_column_idx
                else:
                    features[i][2+j]=node.columns[j]
            
            l=len(node.join_tables)
            for j in range(3):
                if j>=l:
                    features[i][4+j]=self.tablesid2idx[None]
                elif node.join_tables[j]=='middle_tabble' or node.join_tables[j]=='notfound':
                    features[i][4+j]=self.tablesid2idx[None]
                else:
                    features[i][4+j]=self.tablesid2idx[node.join_tables[j]]
            l=len(node.join_columns)
            for j in range(3):
                if j>=l:
                    features[i][7+j]=self.none_column_idx
                else:
                    features[i][7+j]=node.join_columns[j]

            if plan.is_act:
                features[i][10]=node.act_cost
                features[i][11]=node.act_rows
            else:
                features[i][10]=node.est_cost
                features[i][11]=node.est_rows

        
        costmax=np.max(features[:,10])
        costmin=np.min(features[:,10])
        cardmax=np.max(features[:,11])
        cardmin=np.min(features[:,11])
        costabs=(costmax-costmin)
        cardabs=(cardmax-cardmin)
        if costabs==0:
            features[:,10]=1
        else:
            features[:,10]=(features[:,10]-costmin)/costabs
        
        if cardabs==0:
            features[:,11]=1
        else:
            features[:,11]=(features[:,11]-cardmin)/cardabs
        
        return {'features':features,'attention_bias':plan.adj,'heights':plan.heights,'indexes_features':indexes_features,'join_schema_bias':self.join_schema_bias,'length':len(plan.dfs_nodes),'index_length':index_length}

    def get_index_features(self,new_indexes):
        new_indexes=list(new_indexes)
        new_indexes_idx=[]
        for indexes in new_indexes:
            names=indexes._column_names()
            new_indexes_idx.append([self.pWorkload.indexable_columns_dict[name] for name in names])
        
        indexes_features=[]
        for indexes in new_indexes_idx:
            idf=[0,0]
            for i,idx in enumerate(indexes):
                idf[i]=idx+1
            indexes_features.append(idf)
        indexes_features=np.array(indexes_features,dtype=float)


        return indexes_features
    
    def get_random_encoding(self):
        index_length=random.randint(1,30)
        indexes_features=np.random.randint(0,len(self.pWorkload.indexable_columns_dict)+1,size=(index_length,2))

        plan_length=random.randint(5,24)

        features=np.zeros((plan_length,12),dtype=float)
        for i in range(plan_length):
            features[i][0]=random.randint(0,len(self.node_types2idx)-1)
            features[i][1]=random.randint(0,len(self.tablesid2idx)-1)
            for j in range(2):
                features[i][2+j]=random.randint(0,len(self.pWorkload.indexable_columns_dict)-1)
            
            for j in range(3):
                features[i][4+j]=random.randint(0,len(self.tablesid2idx)-1)
            for j in range(3):
                features[i][7+j]=random.randint(0,len(self.pWorkload.indexable_columns_dict)-1)

            features[i][10]=random.random()
            features[i][11]=random.random()
        heights=np.random.randint(0,int(plan_length/2),size=plan_length)
        adj=np.random.randint(0,plan_length,size=(plan_length,plan_length))
        heights=heights.astype(float)
        adj=adj.astype(float)
        return {'features':features,'attention_bias':adj,'heights':heights,'indexes_features':indexes_features,'join_schema_bias':self.join_schema_bias,'length':plan_length,'index_length':index_length}

        
    def column_in_idxs(self,columns,indexes_idx):
        columns_set=set(columns)
        for idx in indexes_idx:
            if columns_set<=set(idx):
                return idx
        return None

    def index_change_column(self,node,new_indexes_idx):
        node_type_id=phy2log_map[node.nodeType]
        can_idexed=True
        index_idxs=None
        if node_type_id==0 or node_type_id==6 or node_type_id==1:
            for column in node.columns:
                if self.pWorkload.indexable_columns[column].column_type not in can_index_column_tpye:
                    can_idexed=False
            for op in node.operatores:
                if op not in can_index_operator_str:
                    can_idexed=False
            index_idxs=self.column_in_idxs(node.columns,new_indexes_idx)
            if index_idxs is None:
                can_idexed=False
        else:
            can_idexed=False
        
        if node.has_or and len(node.columns)>1:
            can_idexed=False
        if can_idexed:
            return index_idxs
        else:
            return None

class LIBDataset(Dataset):
    def __init__(self, plan_cache:dict,query_text2query,split_path:str):
        super(LIBDataset, self).__init__()
        self.plan_cache=plan_cache
        self.queries_2_index_plan_dict={}
        self.queries_2_noindex_plan_dict={}

        self.collated_dicts=[]
        
        self.queries=[]
        self.indexes=[]
        self.plans=[]
        self.act_times=[]
        for k,v in plan_cache.items():
            self.queries.append(query_text2query[k[0]])
            self.indexes.append(k[1])
            self.plans.append(v)
            self.act_times.append(v['Actual Total Time'])
        self.pWorkload=ProcessedWorkload(constants.config)
        self.pWorkload.add_from_triple(self.queries,self.indexes,self.plans)
        
        if type(list(self.indexes[0])[0])==str:
            print(f'indexes is str, need to convert to Index')
            indexes=[]
            for index in self.indexes:
                indexes.append(self._indexes_str2indexes(index))
            self.indexes=indexes
            self.pWorkload.indexes=indexes
        else :
            print(f'indexes is Index, no need to convert to Index')
        
        queries_set=set(self.queries)
        indexes_set=set(self.indexes)
        self.split_data_info=load_checkpoint(split_path)
        
        self.encoding=Encoding(self.pWorkload)
        for query,index,pPlan,act_time in zip(self.queries,self.indexes,self.pWorkload.processed_plans,self.act_times):
            if query in self.queries_2_index_plan_dict:
                self.queries_2_index_plan_dict[query].append((index,pPlan,act_time))
            else:
                self.queries_2_index_plan_dict[query]=[(index,pPlan,act_time)]
            if len(index)==0:
                self.queries_2_noindex_plan_dict[query]=(index,pPlan,act_time)


    def generate_data_with_OOD(self,drop_threshold=None):

        self.data_collated_dicts={'train':[],'test':[],'ood':[],'random':[]}
        self.data_collated_info_dicts={'train':[],'test':[],'ood':[],'random':[]}

        for query,ips in self.queries_2_index_plan_dict.items():
            no_index,no_plan,no_time=self.queries_2_noindex_plan_dict[query]
            for ip in ips:
                collated=self.encoding.get_plan_encoder(no_plan,ip[0])
                ratio=(no_time-ip[2])/no_time
                real=ip[2]
                if drop_threshold is not None and ratio<drop_threshold:
                    continue
                classes=self._get_classes(query,ip[0])
                self.data_collated_dicts[classes].append((collated,ratio,real))
                self.data_collated_info_dicts[classes].append((query,ip[0]))
                
                self.collated_dicts.append((collated,ratio,real))
        
        for i in range(100):
            collated=self.encoding.get_random_encoding()
            self.data_collated_dicts['random'].append((collated,0,0))
            self.data_collated_info_dicts['random'].append((None,None))
            self.collated_dicts.append((collated,None,None))
        for k,v in self.data_collated_dicts.items():
            print(f'{k} has {len(v)} data')
        return self.data_collated_dicts

    def _get_classes(self,query,index):
        if query.nr in self.split_data_info['ood_query']:
            return 'ood'
        elif (query.nr,self._frozenset_to_str(index)) in self.split_data_info['test_query_index']:
            return 'test'
        else:
            return 'train'

    def _indexes_str2indexes(self,indexes_str_list):
        indexes_config=[]
        for indexes_str in indexes_str_list:

            columns_str=re.findall(r'\.(\w+)',indexes_str)
            columns=[]
            for column_str in columns_str:
                i=self.pWorkload.indexable_columns_dict[column_str]
                columns.append(self.pWorkload.indexable_columns[i])
            indexes_config.append(Index(columns))
            
        return frozenset(indexes_config)

    def _frozenset_to_str(self,index_frozenset):
        indexes=[str(index) for index in index_frozenset]
        indexes.sort()
        return ','.join(indexes)
    
    def shuffle(self):
        for k,v in self.data_collated_dicts.items():
            seed=random.randint(0,100)
            np.random.seed(seed)
            np.random.shuffle(v)
            np.random.seed(seed)
            np.random.shuffle(self.data_collated_info_dicts[k])
            
    
    def dump_collated_dicts(self,path):
        save2checkpoint(obj=self.data_collated_dicts,path=path)
        save2checkpoint(obj=self.data_collated_info_dicts,path=path[:-4]+'_info'+path[-4:])

    def get_what_if_cost(self,cost_evaluation,drop_threshold=-8):
        from selection.workload import Workload
        self.what_if_data={'train':[],'test':[],'ood':[]}
        no_index_cost={}
        for query,ips in self.queries_2_index_plan_dict.items():
            no_index,no_plan,no_time=self.queries_2_noindex_plan_dict[query]
            for ip in ips:
                ratio=(no_time-ip[2])/no_time
                real=ip[2]
                if ratio<drop_threshold:
                    continue
                if query not in no_index_cost:
                    origin_cost=cost_evaluation.calculate_cost(Workload([query]),[])
                    no_index_cost[query]=origin_cost
                else:
                    origin_cost=no_index_cost[query]
                after_cost=cost_evaluation.calculate_cost(Workload([query]),list(ip[0]))
                pred_ratio=(origin_cost-after_cost)/origin_cost
                data=(ratio,pred_ratio,real,no_time*((1-pred_ratio)))

                classes=self._get_classes(query,ip[0])
                self.what_if_data[classes].append(data)
        
        return self.what_if_data
    def dump_what_if_data(self,path):
        save2checkpoint(path=path,obj=self.what_if_data)

