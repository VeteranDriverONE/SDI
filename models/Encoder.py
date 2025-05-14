import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

full_atom_feature_dims = get_atom_feature_dims() # 原子特征总的属性数量
full_bond_feature_dims = get_bond_feature_dims() # 键特征总的属性数量

class AtomEncoder(torch.nn.Module):
    """该类用于对原子属性做嵌入。
    记`N`为原子属性的维度，则原子属性表示为`[x1, x2, ..., xi, xN]`，其中任意的一维度`xi`都是类别型数据。full_atom_feature_dims[i]存储了原子属性`xi`的类别数量。
    该类将任意的原子属性`[x1, x2, ..., xi, xN]`转换为原子的嵌入`x_embedding`（维度为emb_dim）。
    """
    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)  # 不同维度的属性用不同的Embedding方法，将dim个离散类编转换为离散值
            torch.nn.init.xavier_uniform_(emb.weight.data) # 初始化Embed参数
            self.atom_embedding_list.append(emb) 

    def forward(self, x):
        # x shape: 原子数，属性维数
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i]) # 将原子每一维特征，经embedding编码后相加

        return x_embedding


class BondEncoder(torch.nn.Module):
    
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()
        
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:,i]) # 将键每一维特征，经embedding编码后相加

        return bond_embedding 
