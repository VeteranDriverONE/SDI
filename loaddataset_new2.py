import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from preprocess_new import create_pytorch_geometric_graph_data_list_from_smiles_and_labels2


def load_feature_smiles(dataset_path, label_flag='classification', vis=False, out_weight=False):
    if dataset_path[-3:] == 'tsv':
        data_file = pd.read_csv(dataset_path, sep='\t')
    elif dataset_path[-3:] == 'csv':
        data_file = pd.read_csv(dataset_path)
    else:
        assert False, f'{dataset_path}:未知的后缀'
    drug_ls = pd.read_csv('dataset/selected_drugs_smiles1.tsv', sep='\t')
    son_ls = pd.read_csv('dataset/selected_excipients_smiles1.tsv', sep='\t')
    for i in range(len(drug_ls)):
        name = drug_ls.loc[i, 'NAME']
        smiles = drug_ls.loc[i, 'SMILES']
    #     Draw.MolToFile(Chem.MolFromSmiles(smiles), f'molpng/{name}.png', size=(600, 600))
    # for i in range(len(son_ls)):
    #     name = son_ls.loc[i, 'NAME']
    #     smiles = son_ls.loc[i, 'SMILES']
    #     Draw.MolToFile(Chem.MolFromSmiles(smiles), f'molpng/{name}.png', size=(600, 600))

    row, _ = data_file.shape
    A_ls = []
    B_ls = []
    label_ls = []
    S2D_weights = []
    D2S_weights = []
    
    for i in range(row):
        A_ls.append(data_file.iloc[i,1])
        B_ls.append(data_file.iloc[i,2])
        if label_flag == 'regression':
            label_ls.append(data_file.iloc[i,3])
        elif label_flag == 'classification':
            label_ls.append(data_file.iloc[i,4])
            tmp_lab = data_file[data_file['sonosensitizer']==data_file.iloc[i,1]]['classes']
            S2D_weights.append(1-(data_file.iloc[i,4]==tmp_lab).sum() / len(tmp_lab)+1e-2)
            tmp_lab = data_file[data_file['drug']==data_file.iloc[i,2]]['classes']
            D2S_weights.append(1-(data_file.iloc[i,4]==tmp_lab).sum() / len(tmp_lab)+1e-2)
        else:
            assert False, '未实现'

    # 药物属性
    drug_name_ls = list(drug_ls.iloc[:,0])
    drug_smiles_ls = list(drug_ls.iloc[:,1])
    # drug_node_features, drug_edge_index, drug_adj_mat = ConvMolFeature2NodeFeature(drug_smiles_ls)
    # node_features, edge_features =  WeaveFeaturizer2NodeFeature(drug_smiles_ls)
    drug_node_features, drug_edge_index, drug_edge_feature, drug_adj_mat = create_pytorch_geometric_graph_data_list_from_smiles_and_labels2(drug_smiles_ls)
    drug_dict = dict(zip(drug_name_ls,list(zip(drug_node_features, drug_edge_index, drug_adj_mat, drug_edge_feature, drug_smiles_ls))))
    
    # 声敏剂属性
    son_name_ls = list(son_ls.iloc[:,0])
    son_smiles_ls = list(son_ls.iloc[:,1])
    # son_node_features, son_edge_index, son_adj_mat = ConvMolFeature2NodeFeature(son_smiles_ls)
    son_node_features, son_edge_index, son_edge_feature, son_adj_mat = create_pytorch_geometric_graph_data_list_from_smiles_and_labels2(son_smiles_ls)
    # node_features, edge_features =  WeaveFeaturizer2NodeFeature(son_smiles_ls)
    son_dict = dict(zip(son_name_ls,list(zip(son_node_features, son_edge_index, son_adj_mat, son_edge_feature, son_smiles_ls))))
    
    drug_features_ls = []
    drug_edge_index_ls = []
    drug_adj_ls = []
    drug_edge_features_ls = []

    son_features_ls = []
    son_edge_index_ls = []
    son_adj_ls = []
    son_edge_features_ls = []

    drug_info = []
    son_info = []

    for son_name, drug_name in zip(A_ls, B_ls):
        drug_features_ls.append(drug_dict[drug_name][0])
        drug_edge_index_ls.append(drug_dict[drug_name][1])
        drug_adj_ls.append(drug_dict[drug_name][2])
        drug_edge_features_ls.append(drug_dict[drug_name][3])

        son_features_ls.append(son_dict[son_name][0])
        son_edge_index_ls.append(son_dict[son_name][1])
        son_adj_ls.append(son_dict[son_name][2])
        son_edge_features_ls.append(son_dict[son_name][3])

        drug_info.append({'drug-name':drug_name,'drug-smile':drug_dict[drug_name][4]})
        son_info.append({'son-name':son_name, 'son-smile':son_dict[son_name][4]})

    label_onehot = np.zeros((len(label_ls), 2))
    for i, val in enumerate(label_ls):
        label_onehot[i, val] = 1

    if not vis and not out_weight:
        return drug_features_ls, drug_edge_index_ls, drug_adj_ls,  drug_edge_features_ls, son_features_ls, son_edge_index_ls, son_adj_ls, son_edge_features_ls, label_onehot
    elif vis and not out_weight:
        return drug_features_ls, drug_edge_index_ls, drug_adj_ls,  drug_edge_features_ls, son_features_ls, son_edge_index_ls, son_adj_ls, son_edge_features_ls, label_onehot, drug_info, son_info
    elif not vis and out_weight:
        return drug_features_ls, drug_edge_index_ls, drug_adj_ls,  drug_edge_features_ls, son_features_ls, son_edge_index_ls, son_adj_ls, son_edge_features_ls, label_onehot, D2S_weights, S2D_weights
    else:
        return drug_features_ls, drug_edge_index_ls, drug_adj_ls,  drug_edge_features_ls, son_features_ls, son_edge_index_ls, son_adj_ls, son_edge_features_ls, label_onehot, D2S_weights, S2D_weights,  drug_info, son_info


def load_feature_smiles_2(dataset_path, label_flag='classification', norm_size=True, label_norm_size=False, son_dict_path=None, drug_dict_path=None):
    # 包含粒径

    if dataset_path[-3:] == 'tsv':
        data_file = pd.read_csv(dataset_path, sep='\t')
    elif dataset_path[-3:] == 'csv':
        data_file = pd.read_csv(dataset_path)
    else:
        assert False, f'{dataset_path}:未知的后缀'
        
    # drug_ls = pd.read_csv('dataset/drugbank/task3/new3/5fold_task3/selected_drugs_smiles_val.csv', sep=',')
    # son_ls = pd.read_csv('dataset/drugbank/task3/new3/5fold_task3/selected_excipients_smiles_val.csv', sep=',')
    drug_ls = pd.read_csv(drug_dict_path, sep=',')
    son_ls = pd.read_csv(son_dict_path, sep=',')
    # drug_ls = pd.read_csv(args.drug_dict, sep='\t')
    # son_ls = pd.read_csv(args.son_dict, sep='\t')


    for i in range(len(drug_ls)):
        name = drug_ls.loc[i, 'NAME']
        smiles = drug_ls.loc[i, 'SMILES']
        size = drug_ls.loc[i, 'SIZE']
    #     Draw.MolToFile(Chem.MolFromSmiles(smiles), f'molpng/{name}.png', size=(600, 600))

    # for i in range(len(son_ls)):
    #     name = son_ls.loc[i, 'NAME']
    #     smiles = son_ls.loc[i, 'SMILES']
    #     Draw.MolToFile(Chem.MolFromSmiles(smiles), f'molpng/{name}.png', size=(600, 600))

    row, _ = data_file.shape
    A_ls = []
    B_ls = []
    label_ls = []
    S2D_weights = []
    D2S_weights = []
    
    for i in range(row):
        A_ls.append(data_file.iloc[i,1]) # son
        B_ls.append(data_file.iloc[i,2]) # drug
        if label_flag == 'regression':
            label_ls.append(data_file.iloc[i,3])
            S2D_weights.append(1)
            D2S_weights.append(1)
        elif label_flag == 'classification':
            label_ls.append(data_file.iloc[i,4])
            tmp_lab = data_file[data_file['sonosensitizer']==data_file.iloc[i,1]]['classes']
            S2D_weights.append(1-(data_file.iloc[i,4]==tmp_lab).sum() / len(tmp_lab)+1e-2)

            tmp_lab = data_file[data_file['drug']==data_file.iloc[i,2]]['classes']
            D2S_weights.append(1-(data_file.iloc[i,4]==tmp_lab).sum() / len(tmp_lab)+1e-2)
        else:
            assert False, '未实现'

    # 药物属性
    drug_name_ls = list(drug_ls.iloc[:,0])
    drug_smiles_ls = list(drug_ls.iloc[:,1])
    drug_size_ls = list(drug_ls.iloc[:,2])
    
    # 声敏剂属性
    son_name_ls = list(son_ls.iloc[:,0])
    son_smiles_ls = list(son_ls.iloc[:,1])
    son_size_ls = list(son_ls.iloc[:,2])

    if norm_size != False:
        size_np = np.array(drug_size_ls+son_size_ls)
        drug_num = len(drug_size_ls)
        
        if isinstance(norm_size, tuple):
            assert len(norm_size) == 2, '长度应为2, 表示均值和标准差'
            m = norm_size[0]
            std = norm_size[1]
        elif isinstance(norm_size, bool) and norm_size:
            m = size_np.mean()
            std = size_np.std()

        new_size_np = (size_np - m)/std 
        new_size = new_size_np.tolist()
        drug_size_ls = new_size[:drug_num]
        son_size_ls = new_size[drug_num:]
        norm_size = (m, std)
    else:
        label_norm_size=False

    # son_node_features, son_edge_index, son_adj_mat = ConvMolFeature2NodeFeature(son_smiles_ls)
    son_node_features, son_edge_index, son_edge_feature, son_adj_mat = create_pytorch_geometric_graph_data_list_from_smiles_and_labels2(son_smiles_ls, son_size_ls)
    # son_node_features, son_edge_index, son_edge_feature, son_adj_mat = create_pytorch_geometric_graph_data_list_from_smiles_and_labels2(son_smiles_ls)
    # node_features, edge_features =  WeaveFeaturizer2NodeFeature(son_smiles_ls)
    son_dict = dict(zip(son_name_ls,list(zip(son_node_features, son_edge_index, son_adj_mat, son_edge_feature, son_smiles_ls, son_size_ls))))
    # son_dict = dict(zip(son_name_ls,list(zip(son_node_features, son_edge_index, son_adj_mat, son_edge_feature, son_smiles_ls))))

    # drug_node_features, drug_edge_index, drug_adj_mat = ConvMolFeature2NodeFeature(drug_smiles_ls)
    # node_features, edge_features =  WeaveFeaturizer2NodeFeature(drug_smiles_ls)
    drug_node_features, drug_edge_index, drug_edge_feature, drug_adj_mat = create_pytorch_geometric_graph_data_list_from_smiles_and_labels2(drug_smiles_ls, drug_size_ls)
    drug_dict = dict(zip(drug_name_ls,list(zip(drug_node_features, drug_edge_index, drug_adj_mat, drug_edge_feature, drug_smiles_ls, drug_size_ls))))
    
    # drug_node_features, drug_edge_index, drug_edge_feature, drug_adj_mat = create_pytorch_geometric_graph_data_list_from_smiles_and_labels2(drug_smiles_ls)
    # drug_dict = dict(zip(drug_name_ls,list(zip(drug_node_features, drug_edge_index, drug_adj_mat, drug_edge_feature, drug_smiles_ls))))
    

    drug_features_ls = []
    drug_edge_index_ls = []
    drug_adj_ls = []
    drug_edge_features_ls = []

    son_features_ls = []
    son_edge_index_ls = []
    son_adj_ls = []
    son_edge_features_ls = []

    drug_info = []
    son_info = []

    for son_name, drug_name in zip(A_ls, B_ls):
        if son_dict.get(son_name) is None or drug_dict.get(drug_name) is None:
            print(f'{son_name} or {drug_name} is None')
            continue
        son_features_ls.append(son_dict[son_name][0])
        son_edge_index_ls.append(son_dict[son_name][1])
        son_adj_ls.append(son_dict[son_name][2])
        son_edge_features_ls.append(son_dict[son_name][3])

        drug_features_ls.append(drug_dict[drug_name][0])
        drug_edge_index_ls.append(drug_dict[drug_name][1])
        drug_adj_ls.append(drug_dict[drug_name][2])
        drug_edge_features_ls.append(drug_dict[drug_name][3])

        son_info.append({'son-name':son_name, 'son-smile':son_dict[son_name][4],'son-size':son_dict[son_name][5]})
        drug_info.append({'drug-name':drug_name,'drug-smile':drug_dict[drug_name][4],'drug-size':drug_dict[drug_name][5]})
        # son_info.append({'son-name':son_name, 'son-smile':son_dict[son_name][4]})
        # drug_info.append({'drug-name':drug_name,'drug-smile':drug_dict[drug_name][4]})
        

    if label_flag == 'regression':
        label_ls = np.array(label_ls).reshape((len(label_ls), 1))
        label_ls = (label_ls-m)/std if label_norm_size else label_ls
    else:
        label_onehot = np.zeros((len(label_ls), 2))
        for i, val in enumerate(label_ls):
            label_onehot[i, val] = 1
            label_ls = label_onehot

    return son_features_ls, son_edge_index_ls, son_adj_ls, son_edge_features_ls, drug_features_ls, drug_edge_index_ls, drug_adj_ls,  drug_edge_features_ls,  label_ls,  S2D_weights,  D2S_weights, son_info, drug_info, norm_size

class GraphDataset(Dataset):
    def __init__(self, node_feature:list, edge_index:list, label:list, **args):
        super(GraphDataset, self).__init__()
        # node, edge_index, label
        self.data_list = []
        for i in range(len(node_feature)):
            tmp_data = Data(x=torch.tensor(node_feature[i], dtype = torch.float32),
                            edge_index=torch.tensor(edge_index[i], dtype=torch.float32), 
                            y=torch.tensor(label[i,:], dtype=torch.int8))
            for key, val in args.items():
                setattr(tmp_data,key,torch.tensor(val[i],dtype=torch.float32))
            self.data_list.append(tmp_data)
        print('load finished')

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        return data
    
class GraphDataset2(Dataset):
    def __init__(self, dataset_path, label_flag='classification', adj=False, bidrection=False):
        super(GraphDataset2, self).__init__()
        # node, edge_index, label
        node_attr1, edge_index1, edge_adj1, edge_attr1, node_attr2, edge_index2, edge_adj2, edge_attr2, label= load_feature_smiles(dataset_path, label_flag)
        self.data_list1 = []
        self.data_list2 = []
        for i in range(len(node_attr1)):
            if adj:
                tmp_data1 = Data(x=torch.tensor(node_attr1[i], dtype = torch.float32),
                                edge_index=torch.tensor(edge_adj1[i], dtype=torch.int64), 
                                y=torch.tensor(label[i,:]))
                tmp_data2 = Data(x=torch.tensor(node_attr2[i], dtype = torch.float32),
                                edge_index=torch.tensor(edge_adj2[i], dtype=torch.int64), 
                                y=torch.tensor(label[i,:]))
            else:
                tmp_data1 = Data(x=torch.tensor(node_attr1[i], dtype = torch.float32),
                                edge_index=torch.tensor(edge_index1[i], dtype=torch.int64), 
                                y=torch.tensor(label[i,:]))
                tmp_data2 = Data(x=torch.tensor(node_attr2[i], dtype = torch.float32),
                                edge_index=torch.tensor(edge_index2[i], dtype=torch.int64), 
                                y=torch.tensor(label[i,:]))
            if edge_attr1 is not None and edge_attr2 is not None:
                tmp_data1.edge_attr = edge_attr1[i]
                tmp_data2.edge_attr = edge_attr2[i]
            self.data_list1.append(tmp_data1)
            self.data_list2.append(tmp_data2)
            # 反向，药物量翻倍
            if bidrection:
                self.data_list1.append(tmp_data2)
                self.data_list2.append(tmp_data1)
        print('load finished')

    def __len__(self):
        return len(self.data_list1)
    
    def __getitem__(self, idx):
        data1 = self.data_list1[idx]
        data2 = self.data_list2[idx]
        edge_node1 = Data(x=data1.edge_attr, edge_index=torch.LongTensor())
        edge_node2 = Data(x=data2.edge_attr, edge_index=torch.LongTensor())
        return data1, data2, edge_node1, edge_node2
    

class GraphDataset3(Dataset):
    def __init__(self, dataset_path, label_flag='classification', adj=False, bidrection=False):
        super(GraphDataset3, self).__init__()
        # node, edge_index, label
        node_attr1, edge_index1, edge_adj1, edge_attr1, node_attr2, edge_index2, edge_adj2, edge_attr2, label, drug_info, son_info = load_feature_smiles(
            dataset_path, label_flag, vis=True)
        self.data_list1 = []
        self.data_list2 = []
        self.drug_info = []
        for i in range(len(node_attr1)):
            if adj:
                tmp_data1 = Data(x=node_attr1[i], edge_index=edge_adj1[i], 
                                y=torch.tensor(label[i,:]))
                tmp_data2 = Data(x=node_attr2[i], edge_index=edge_adj2[i],  
                                y=torch.tensor(label[i,:]))
                
            else:
                tmp_data1 = Data(x=node_attr1[i], edge_index=edge_index1[i],  
                                y=torch.tensor(label[i,:]))
                tmp_data2 = Data(x=node_attr2[i], edge_index=edge_index2[i], 
                                y=torch.tensor(label[i,:]))
            
            if edge_attr1 is not None and edge_attr2 is not None:
                tmp_data1.edge_attr = edge_attr1[i]
                tmp_data2.edge_attr = edge_attr2[i]              
            self.data_list1.append(tmp_data1)
            self.data_list2.append(tmp_data2)
            self.drug_info.append({'drug1-name':drug_info[i]['drug-name'],'drug1-smile':drug_info[i]['drug-smile'],
                                   'drug2-name':son_info[i]['son-name'],'drug2-smile':son_info[i]['son-smile']})
            # 反向
            if bidrection:
                self.data_list1.append(tmp_data2)
                self.data_list2.append(tmp_data1)
                self.drug_info.append({'drug1-name':son_info[i]['son-name'], 'drug1-smile':son_info[i]['son-smile'],
                                    'drug2-name':drug_info[i]['drug-name'], 'drug2-smile':drug_info[i]['drug-smile']})
        print('load finished')

    def __len__(self):
        return len(self.data_list1)
    
    def __getitem__(self, idx):
        data1 = self.data_list1[idx]
        data2 = self.data_list2[idx]
        drug_info = self.drug_info[idx]
        edge_node1 = Data(x=data1.edge_attr, edge_index=torch.LongTensor())
        edge_node2 = Data(x=data2.edge_attr, edge_index=torch.LongTensor())
        return data1, data2, edge_node1, edge_node2, drug_info
    

class GraphDataset4(Dataset):
    def __init__(self, dataset_path, label_flag='classification', adj=False, bidrection=False):
        super(GraphDataset4, self).__init__()
        # node, edge_index, label
        node_attr1, edge_index1, edge_adj1, edge_attr1, node_attr2, edge_index2, edge_adj2, edge_attr2, label, weights_D2S, weights_S2D, drug_info, son_info = load_feature_smiles(
            dataset_path, label_flag, out_weight=True, vis=True)
        self.data_list1 = []
        self.data_list2 = []
        self.weights = []
        self.drug_info = []
        for i in range(len(node_attr1)):
            if adj:
                tmp_data1 = Data(x=torch.tensor(node_attr1[i], dtype = torch.float32),
                                edge_index=torch.tensor(edge_adj1[i], dtype=torch.int64), 
                                y=torch.tensor(label[i,:]))
                tmp_data2 = Data(x=torch.tensor(node_attr2[i], dtype = torch.float32),
                                edge_index=torch.tensor(edge_adj2[i], dtype=torch.int64), 
                                y=torch.tensor(label[i,:]))
            else:
                tmp_data1 = Data(x=node_attr1[i].float(),
                                edge_index=edge_index1[i].long(), 
                                y=torch.tensor(label[i,:]))
                tmp_data2 = Data(x=node_attr2[i].float(),
                                edge_index=edge_index2[i].long(), 
                                y=torch.tensor(label[i,:]))
            if edge_attr1 is not None and edge_attr2 is not None:
                tmp_data1.edge_attr = edge_attr1[i]
                tmp_data2.edge_attr = edge_attr2[i]
            self.data_list1.append(tmp_data1)
            self.data_list2.append(tmp_data2)
            self.weights.append(weights_S2D[i])
            self.drug_info.append({'drug1-name':son_info[i]['son-name'], 'drug1-smile':son_info[i]['son-smile'],
                                    'drug2-name':drug_info[i]['drug-name'], 'drug2-smile':drug_info[i]['drug-smile']})
            # 反向，药物量翻倍
            if bidrection:
                self.data_list1.append(tmp_data2)
                self.data_list2.append(tmp_data1)
                self.weights.append(weights_D2S[i])
                self.drug_info.append({'drug1-name':son_info[i]['son-name'], 'drug1-smile':son_info[i]['son-smile'],
                                    'drug2-name':drug_info[i]['drug-name'], 'drug2-smile':drug_info[i]['drug-smile']})
        print('load finished')

    def __len__(self):
        return len(self.data_list1)
    
    def __getitem__(self, idx):
        data1 = self.data_list1[idx]
        data2 = self.data_list2[idx]
        edge_node1 = Data(x=data1.edge_attr, edge_index=torch.LongTensor())
        edge_node2 = Data(x=data2.edge_attr, edge_index=torch.LongTensor())
        weight = self.weights[idx]
        drug_info = self.drug_info[idx]
        return data1, data2, edge_node1, edge_node2, weight, drug_info


class GraphDataset5(Dataset):
    def __init__(self, dataset_path:str, args, label_flag='classification', adj=False, bidrection=False, norm_size=True, label_norm_size=False):
        super(GraphDataset5, self).__init__()
        # node, edge_index, label
        adj = args.use_adj
        node_attr1, edge_index1, edge_adj1, edge_attr1, node_attr2, edge_index2, edge_adj2, edge_attr2, label, weights_S2D, weights_D2S, son_info, drug_info, norm_size \
                    = load_feature_smiles_2(dataset_path, label_flag, norm_size=norm_size, label_norm_size=label_norm_size, 
                                            son_dict_path=args.son_dict, drug_dict_path=args.drug_dict)
        self.norm_size = norm_size
        self.data_list1 = []
        self.data_list2 = []
        self.data_list = []
        self.weights = []
        self.drug_info = []
        for i in range(len(node_attr1)):
            if adj:
                tmp_data1 = Data(x=torch.tensor(node_attr1[i], dtype = torch.float32),
                                edge_index=torch.tensor(edge_adj1[i], dtype=torch.int64), 
                                y=torch.tensor(label[i,:]))
                
                tmp_data2 = Data(x=torch.tensor(node_attr2[i], dtype = torch.float32),
                                edge_index=torch.tensor(edge_adj2[i], dtype=torch.int64), 
                                y=torch.tensor(label[i,:]))
            else:
                tmp_data1 = Data(x=node_attr1[i].float(),
                                edge_index=edge_index1[i].long(), 
                                y=torch.tensor(label[i,:]))
                
                tmp_data2 = Data(x=node_attr2[i].float(),
                                edge_index=edge_index2[i].long(), 
                                y=torch.tensor(label[i,:]))
                
            if edge_attr1 is not None and edge_attr2 is not None:
                tmp_data1.edge_attr = edge_attr1[i]
                tmp_data2.edge_attr = edge_attr2[i]
            self.data_list1.append(tmp_data1)
            self.data_list2.append(tmp_data2)
            self.weights.append(weights_S2D[i])
            self.drug_info.append({'drug1-name':son_info[i]['son-name'], 'drug1-smile':son_info[i]['son-smile'],'drug1-size':son_info[i]['son-size'],
                                    'drug2-name':drug_info[i]['drug-name'], 'drug2-smile':drug_info[i]['drug-smile'],'drug2-size':drug_info[i]['drug-size']})
            # self.drug_info.append({'drug1-name':son_info[i]['son-name'], 'drug1-smile':son_info[i]['son-smile'],
            #                         'drug2-name':drug_info[i]['drug-name'], 'drug2-smile':drug_info[i]['drug-smile']})
            # 反向，药物量翻倍
            if bidrection:
                self.data_list1.append(tmp_data2)
                self.data_list2.append(tmp_data1)
                self.weights.append(weights_D2S[i])
                self.drug_info.append({'drug1-name':drug_info[i]['drug-name'], 'drug1-smile':drug_info[i]['drug-smile'],'drug1-size':drug_info[i]['drug-size'],
                                    'drug2-name':son_info[i]['son-name'], 'drug2-smile':son_info[i]['son-smile'],'drug2-size':son_info[i]['son-size']})
                # self.drug_info.append({'drug1-name':drug_info[i]['drug-name'], 'drug1-smile':drug_info[i]['drug-smile'],
                #                     'drug2-name':son_info[i]['son-name'], 'drug2-smile':son_info[i]['son-smile']})            
        print('load finished')

    def __len__(self):
        return len(self.data_list1)
    
    def __getitem__(self, idx):
        data1 = self.data_list1[idx]
        data2 = self.data_list2[idx]
        edge_node1 = Data(x=data1.edge_attr, edge_index=torch.LongTensor())
        edge_node2 = Data(x=data2.edge_attr, edge_index=torch.LongTensor())
        weight = self.weights[idx]
        drug_info = self.drug_info[idx]
        return data1, data2, edge_node1, edge_node2, weight, drug_info
        ######Draw drug_vis
        # return data1, data2, edge_node1, edge_node2, drug_info