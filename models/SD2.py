import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import GATConv, SAGPooling, LayerNorm, global_add_pool, Linear, Set2Set, TransformerConv, global_mean_pool
from einops import rearrange
from rdkit import Chem
from rdkit.Chem import Draw
import time

class Encode_Block(nn.Module):
    def __init__(self, in_features, n_heads, head_out_feats, edge_feature, dp):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.feature_conv = TransformerConv(in_features, in_features, n_heads, edge_dim=edge_feature,dropout=dp, concat=False)
        self.feature_conv2 = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp, concat=False)
        # self.feature_conv = GATv2Conv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        # self.feature_conv2 = GATv2Conv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        self.norm = LayerNorm(in_features)
        self.norm2 = LayerNorm(head_out_feats)
        self.lin_up = Linear(64, 64, bias=True, weight_initializer='glorot')
        self.lin_up2 = Linear(64, 64, bias=True, weight_initializer='glorot')
    
    def forward(self, drug_data):
        drug_data.x = self.feature_conv(drug_data.x, drug_data.edge_index, drug_data.edge_attr)
        drug_data.x = F.elu(self.norm(drug_data.x, drug_data.batch))
        drug_data.edge_attr = self.lin_up(drug_data.edge_attr)
        drug_data.edge_attr = F.elu(drug_data.edge_attr)

        drug_data.x = self.feature_conv2(drug_data.x, drug_data.edge_index, drug_data.edge_attr)
        drug_data.edge_attr = self.lin_up2(drug_data.edge_attr)
        drug_data.x = F.elu(self.norm2(drug_data.x, drug_data.batch))
        drug_data.edge_attr = F.elu(drug_data.edge_attr)

        return drug_data

class Encode_Block1(nn.Module):
    def __init__(self, in_features, n_heads, head_out_feats, edge_feature, dp):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = TransformerConv(in_features, in_features, n_heads, edge_dim=edge_feature, dropout=dp, concat=False)
        self.feature_conv2 = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature, dropout=dp, concat=False)
        # self.feature_conv = GATv2Conv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        # self.feature_conv2 = GATv2Conv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        self.norm = LayerNorm(in_features)
        self.norm2 = LayerNorm(head_out_feats)
        self.lin_up = Linear(edge_feature, edge_feature, bias=True, weight_initializer='glorot')
        self.lin_up2 = Linear(edge_feature, edge_feature, bias=True, weight_initializer='glorot')
        print(edge_feature)
    
    def forward(self, drug_data):
        drug_data.x = self.feature_conv(drug_data.x, drug_data.edge_index, drug_data.edge_attr)
        drug_data.x = F.elu(self.norm(drug_data.x, drug_data.batch))
        drug_data.edge_attr = self.lin_up(drug_data.edge_attr)
        drug_data.edge_attr = F.elu(drug_data.edge_attr)
        
        drug_data.x = self.feature_conv2(drug_data.x, drug_data.edge_index, drug_data.edge_attr)
        drug_data.edge_attr = self.lin_up2(drug_data.edge_attr)
        drug_data.x = F.elu(self.norm2(drug_data.x, drug_data.batch))
        drug_data.edge_attr = F.elu(drug_data.edge_attr)
        return drug_data
    
class Encode_Block2(nn.Module):
    def __init__(self, in_features, n_heads, head_out_feats, in_edge_feature, out_edge_feature, dp):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = TransformerConv(in_features, in_features, n_heads, edge_dim=in_edge_feature, concat=False)
        self.feature_conv2 = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=in_edge_feature, concat=False)
        # self.feature_conv = GATv2Conv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        # self.feature_conv2 = GATv2Conv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        self.norm = LayerNorm(in_features)
        self.norm2 = LayerNorm(head_out_feats)

        self.lin_up = Linear(in_edge_feature, in_edge_feature, bias=True, weight_initializer='glorot')
        self.lin_up2 = Linear(in_edge_feature, out_edge_feature, bias=True, weight_initializer='glorot')
        # self.norm_e = LayerNorm(in_edge_feature)
        # self.norm_e2 = LayerNorm(out_edge_feature)
        self.norm_e = nn.Dropout(dp)
        self.norm_e2 = nn.Dropout(dp)

    def forward(self, drug_data):
        drug_data.x = self.feature_conv(drug_data.x, drug_data.edge_index, drug_data.edge_attr)
        drug_data.x = F.elu(self.norm(drug_data.x, drug_data.batch))
        drug_data.edge_attr = self.lin_up(drug_data.edge_attr)
        drug_data.edge_attr = F.elu(self.norm_e(drug_data.edge_attr))

        drug_data.x = self.feature_conv2(drug_data.x, drug_data.edge_index, drug_data.edge_attr)
        drug_data.edge_attr = self.lin_up2(drug_data.edge_attr)
        drug_data.x = F.elu(self.norm2(drug_data.x, drug_data.batch))
        drug_data.edge_attr = F.elu(self.norm_e2(drug_data.edge_attr))
        return drug_data


class CoAttentionLayer2(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, query, keyvalue):
        query = self.norm(query)
        keyvalue = self.norm(keyvalue)
        q = self.to_q(query)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        kv = self.to_kv(keyvalue).chunk(2, dim = -1) # 按最后一维划分变成三份
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SD2(nn.Module):
    def __init__(self, nclass, in_node_features, in_edge_features, hidd_dim, n_out_feats, n_heads, edge_feature, dp):
        super(SD2, self).__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.n_out_feats = n_out_feats
        self.hidd_dim = hidd_dim
        self.edge_feature = edge_feature
        self.n_blocks = len(n_heads)
        assert len(self.n_out_feats) == len(n_heads), 'head_out_feats长度与blocks_params必须相等'

        self.initial_node_feature = Linear(self.in_node_features, self.in_node_features ,bias=True, weight_initializer='glorot')
        self.initial_edge_feature = Linear(self.in_edge_features, edge_feature ,bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(self.in_node_features)
        self.n_out_feats .insert(0, self.in_node_features)

        self.encode_blocks = []
        self.readouts = []
        self.to_trans = []
        for i in range(self.n_blocks):
            self.encode_blocks.append(Encode_Block(self.n_out_feats[i], n_heads[i], self.n_out_feats[i+1], edge_feature, dp))
            self.readouts.append(SAGPooling(self.n_out_feats[i+1], min_score=-1)) # min_score，注意力得分大于min_score节点被保留，设置-1表示所有节点做完自注意力后都被保留，或者ratio=1也保留全部节点
            self.to_trans.append(Linear(self.n_out_feats[i+1], hidd_dim, bias=False))
        
        self.n_out_feats = self.n_out_feats [1:]
        self.encode_blocks = nn.ModuleList(self.encode_blocks)
        self.readouts = nn.ModuleList(self.readouts)
        self.to_trans = nn.ModuleList(self.to_trans)
        self.co_attention = CoAttentionLayer2(hidd_dim, self.n_blocks, hidd_dim) 
        self.merge_all = nn.Sequential(nn.Linear(len(self.n_out_feats)**2, nclass))
        self.re_shape_e = Linear(edge_feature, hidd_dim, bias=True, weight_initializer='glorot')

    def forward(self, drug1, drug2, d1_edge, d2_edge):
        # h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels, h_data_edge, t_data_edge = triples
        # h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels = triples
        # 初始维度变换 55-64/128
        drug1.x = self.initial_node_feature(drug1.x) # 转换特征维数
        drug2.x = self.initial_node_feature(drug2.x)
        drug1.x = self.initial_node_norm(drug1.x, drug1.batch) # norm，正则化
        drug2.x = self.initial_node_norm(drug2.x, drug2.batch)
        drug1.x = F.elu(drug1.x) # 非线性激活，
        drug2.x = F.elu(drug2.x)

        drug1.edge_attr = self.initial_edge_feature(drug1.edge_attr)  # 边属性
        drug2.edge_attr = self.initial_edge_feature(drug2.edge_attr)
        drug1.edge_attr = F.elu(drug1.edge_attr)
        drug2.edge_attr = F.elu(drug2.edge_attr)

        # 4层gat 64-32*2
        repr_h = []
        repr_t = []
        for i in range(len(self.encode_blocks)):
            drug1 = self.encode_blocks[i](drug1) # 编码
            drug2 = self.encode_blocks[i](drug2) # 编码
            # readout
            h_global_graph_emb, t_global_graph_emb = self.GlosbalPool(drug1, drug2, d1_edge, d2_edge, i) # 融合节点的全局特征表示
            repr_h.append(h_global_graph_emb)
            repr_t.append(t_global_graph_emb)

        repr_h = torch.stack((repr_h), dim=1) # B, layer num, D
        repr_t = torch.stack((repr_t), dim=1)
        head_attentions = self.co_attention(repr_h, repr_t) # 注意力，heads为key，tails为query, output:B,N,D
        tail_attentions = self.co_attention(repr_t, repr_h)
        out = self.merge_all(torch.matmul(head_attentions, tail_attentions.transpose(-2,-1)).flatten(1)) # B, N*N
        # out = self.co_attention(kge_heads, kge_tails)
        return F.softmax(out, -1)


    def GlosbalPool(self, h_data, t_data, h_data_edge, t_data_edge, i):
        # 节点注意力加权
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readouts[i](h_data.x, h_data.edge_index, edge_attr=h_data.edge_attr, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readouts[i](t_data.x, t_data.edge_index, edge_attr=t_data.edge_attr, batch=t_data.batch)
        # 节点readout
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch) # 节点全局池化, 每个分子图由一个embedding表示;global_add_pol:把所有节点特征表示相加
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)
        #边readout
        h_data_edge.x = h_data.edge_attr
        t_data_edge.x = t_data.edge_attr
        h_global_graph_emb_edge = global_add_pool(h_data_edge.x, batch=h_data_edge.batch) # 边全局池化, B,dim
        t_global_graph_emb_edge = global_add_pool(t_data_edge.x, batch=t_data_edge.batch)
        h_global_graph_emb_edge = F.elu(self.re_shape_e(h_global_graph_emb_edge))
        t_global_graph_emb_edge = F.elu(self.re_shape_e(t_global_graph_emb_edge))
        # h_global_graph_emb = h_global_graph_emb * h_global_graph_emb_edge # 融合图和节点表示
        # t_global_graph_emb = t_global_graph_emb * t_global_graph_emb_edge
        h_global_graph_emb = self.to_trans[i](h_global_graph_emb)
        t_global_graph_emb = self.to_trans[i](t_global_graph_emb)
        h_global_graph_emb = F.normalize(h_global_graph_emb)
        t_global_graph_emb = F.normalize(t_global_graph_emb)
        return h_global_graph_emb, t_global_graph_emb        


class SD21(nn.Module):
    def __init__(self, nclass, in_node_features, in_edge_features, hidd_dim, kge_dim, n_out_feats, n_heads, edge_feature, dp):
        super(SD21, self).__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.n_out_feats = n_out_feats
        self.e_out_feats = n_out_feats.copy()
        self.hidd_dim = hidd_dim
        # self.kge_dim = kge_dim
        self.edge_feature = edge_feature
        self.n_blocks = len(n_heads)
        self.dp = dp
        assert len(n_out_feats) == len(n_heads), 'head_out_feats长度与blocks_params必须相等'
        self.initial_node_feature = Linear(self.in_node_features, self.in_node_features, bias=True, weight_initializer='glorot')
        self.initial_edge_feature = Linear(self.in_edge_features, self.in_edge_features, bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(self.in_node_features)
        self.n_out_feats.insert(0, self.in_node_features)
        self.e_out_feats.insert(0, self.in_edge_features)
        self.encode_blocks = []
        self.readouts = []
        self.to_trans = []
        self.re_shape_e = []
        self.norm_e = []
        for i in range(self.n_blocks):
            self.encode_blocks.append(Encode_Block1(self.n_out_feats[i], n_heads[i], self.n_out_feats[i+1], self.e_out_feats[i], self.dp))
            self.readouts.append(SAGPooling(self.n_out_feats[i+1], min_score=-1)) # min_score，注意力得分大于min_score节点被保留，设置-1表示所有节点做完自注意力后都被保留，或者ratio=1也保留全部节点
            self.to_trans.append(Linear(self.n_out_feats[i+1], hidd_dim, bias=False))
            self.re_shape_e.append(Linear(self.e_out_feats[i], self.e_out_feats[i+1], bias=True, weight_initializer='glorot'))
            self.norm_e.append(LayerNorm(self.e_out_feats[i+1]))
        
        self.n_out_feats.pop(0)
        self.e_out_feats.pop(0)
        self.encode_blocks = nn.ModuleList(self.encode_blocks)
        self.readouts = nn.ModuleList(self.readouts)
        self.to_trans = nn.ModuleList(self.to_trans)
        self.re_shape_e =  nn.ModuleList(self.re_shape_e)
        self.norm_e = nn.ModuleList(self.norm_e)
        self.co_attention = CoAttentionLayer2(hidd_dim, self.n_blocks, hidd_dim) 
        self.merge_all = nn.Sequential(nn.Linear(len(self.n_out_feats)**2, nclass))

    def forward(self, drug1, drug2, d1_edge, d2_edge):
        # h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels, h_data_edge, t_data_edge = triples
        # h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels = triples
        # 初始维度变换 55-64/128
        drug1.x = self.initial_node_feature(drug1.x) # 转换特征维数
        drug2.x = self.initial_node_feature(drug2.x)
        drug1.x = self.initial_node_norm(drug1.x, drug1.batch) # norm，正则化
        drug2.x = self.initial_node_norm(drug2.x, drug2.batch)
        drug1.x = F.elu(drug1.x) # 非线性激活，
        drug2.x = F.elu(drug2.x)
        drug1.edge_attr = self.initial_edge_feature(drug1.edge_attr)  # 边属性
        drug2.edge_attr = self.initial_edge_feature(drug2.edge_attr)
        drug1.edge_attr = F.elu(drug1.edge_attr)
        drug2.edge_attr = F.elu(drug2.edge_attr)
        # 4层gat 64-32*2
        repr_h = []
        repr_t = []
        for i in range(len(self.encode_blocks)):
            drug1 = self.encode_blocks[i](drug1) # 编码
            drug2 = self.encode_blocks[i](drug2) # 编码
            # readout
            h_global_graph_emb, t_global_graph_emb = self.GlosbalPool(drug1, drug2, d1_edge, d2_edge, i) # 融合节点的全局特征表示
            repr_h.append(h_global_graph_emb)
            repr_t.append(t_global_graph_emb)
        repr_h = torch.stack((repr_h), dim=1) # B, layer num, D
        repr_t = torch.stack((repr_t), dim=1)
        head_attentions = self.co_attention(repr_h, repr_t) # 注意力，heads为key，tails为query, output:B,N,D
        tail_attentions = self.co_attention(repr_t, repr_h)
        out = self.merge_all(torch.matmul(head_attentions, tail_attentions.transpose(-2,-1)).flatten(1)) # B, N*N
        # out = self.co_attention(kge_heads, kge_tails)
        return F.softmax(out, -1)


    def GlosbalPool(self, h_data, t_data, h_data_edge, t_data_edge, i):
        # 节点注意力加权
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readouts[i](h_data.x, h_data.edge_index, edge_attr=h_data.edge_attr, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readouts[i](t_data.x, t_data.edge_index, edge_attr=t_data.edge_attr, batch=t_data.batch)
        # 节点readout
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch) # 节点全局池化, 每个分子图由一个embedding表示;global_add_pol:把所有节点特征表示相加
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)
        # 边
        h_attr_edge = F.elu(self.norm_e[i](self.re_shape_e[i](h_data.edge_attr)))
        t_attr_edge = F.elu(self.norm_e[i](self.re_shape_e[i](t_data.edge_attr)))
        h_data.edge_attr = h_attr_edge
        t_data.edge_attr = t_attr_edge
        # 边readout
        h_data_edge.x = h_data.edge_attr
        t_data_edge.x = t_data.edge_attr
        h_global_graph_emb_edge = global_add_pool(h_data_edge.x, batch=h_data_edge.batch) # 边全局池化, B,dim
        t_global_graph_emb_edge = global_add_pool(t_data_edge.x, batch=t_data_edge.batch)
        h_global_graph_emb = h_global_graph_emb * h_global_graph_emb_edge # 融合图和节点表示
        t_global_graph_emb = t_global_graph_emb * t_global_graph_emb_edge
        h_global_graph_emb = self.to_trans[i](h_global_graph_emb)
        t_global_graph_emb = self.to_trans[i](t_global_graph_emb)
        h_global_graph_emb = F.normalize(h_global_graph_emb)
        t_global_graph_emb = F.normalize(t_global_graph_emb)
        return h_global_graph_emb, t_global_graph_emb

class SD22(nn.Module):
    def __init__(self, nclass, in_node_features, in_edge_features, hidd_dim, kge_dim, n_out_feats, n_heads, e_out_feats, dp):
        super(SD22, self).__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.n_out_feats = n_out_feats
        self.e_out_feats = e_out_feats
        self.hidd_dim = hidd_dim
        self.kge_dim = kge_dim
        self.n_blocks = len(n_heads)
        self.dp = dp
        assert len(n_out_feats) == len(n_heads), 'head_out_feats长度与blocks_params必须相等'
        self.initial_node_feature = Linear(self.in_node_features, self.in_node_features, bias=True, weight_initializer='glorot')
        self.initial_edge_feature = Linear(self.in_edge_features, self.in_edge_features, bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(self.in_node_features)
        self.n_out_feats.insert(0, self.in_node_features)
        self.e_out_feats.insert(0, self.in_edge_features)
        self.encode_blocks = []
        self.readouts = []
        self.to_trans = []
        self.re_shape_e = []
        self.norm_e = []
        for i in range(self.n_blocks):
            self.encode_blocks.append(Encode_Block2(self.n_out_feats[i], n_heads[i], self.n_out_feats[i+1], self.e_out_feats[i], self.e_out_feats[i+1], self.dp))
            self.readouts.append(SAGPooling(self.n_out_feats[i+1], min_score=-1)) # min_score，注意力得分大于min_score节点被保留，设置-1表示所有节点做完自注意力后都被保留，或者ratio=1也保留全部节点
            self.to_trans.append(Linear(self.n_out_feats[i+1], hidd_dim, bias=False))
            self.re_shape_e.append(Linear(self.e_out_feats[i+1], self.n_out_feats[i+1], bias=False, weight_initializer='glorot'))
            self.norm_e.append(LayerNorm(self.e_out_feats[i+1]))
        self.n_out_feats.pop(0)
        self.e_out_feats.pop(0)
        self.encode_blocks = nn.ModuleList(self.encode_blocks)
        self.readouts = nn.ModuleList(self.readouts)
        self.to_trans = nn.ModuleList(self.to_trans)
        self.re_shape_e =  nn.ModuleList(self.re_shape_e)
        self.norm_e = nn.ModuleList(self.norm_e)
        self.co_attention = CoAttentionLayer2(hidd_dim, self.n_blocks, hidd_dim) 
        self.merge_all = nn.Sequential(nn.Linear(len(self.n_out_feats)**2, nclass))

    def forward(self, drug1, drug2, d1_edge, d2_edge):
        # h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels, h_data_edge, t_data_edge = triples
        # h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels = triples
        # 初始维度变换 55-64/128
        drug1.x = self.initial_node_feature(drug1.x) # 转换特征维数
        drug2.x = self.initial_node_feature(drug2.x)
        drug1.x = self.initial_node_norm(drug1.x, drug1.batch) # norm，正则化
        drug2.x = self.initial_node_norm(drug2.x, drug2.batch)
        drug1.x = F.elu(drug1.x) # 非线性激活，
        drug2.x = F.elu(drug2.x)
        drug1.edge_attr = self.initial_edge_feature(drug1.edge_attr)  # 边属性
        drug2.edge_attr = self.initial_edge_feature(drug2.edge_attr)
        drug1.edge_attr = F.elu(drug1.edge_attr)
        drug2.edge_attr = F.elu(drug2.edge_attr)
        # 4层gat 64-32*2
        repr_h = []
        repr_t = []
        for i in range(len(self.encode_blocks)):
            drug1 = self.encode_blocks[i](drug1) # 编码
            drug2 = self.encode_blocks[i](drug2) # 编码
            # readout
            h_global_graph_emb, t_global_graph_emb = self.GlosbalPool(drug1, drug2, d1_edge, d2_edge, i) # 融合节点的全局特征表示
            repr_h.append(h_global_graph_emb)
            repr_t.append(t_global_graph_emb)
        repr_h = torch.stack((repr_h), dim=1) # B, layer num, D
        repr_t = torch.stack((repr_t), dim=1)
        head_attentions = self.co_attention(repr_h, repr_t) # 注意力，heads为key，tails为query, output:B,N,D
        tail_attentions = self.co_attention(repr_t, repr_h)
        out = self.merge_all(torch.matmul(head_attentions, tail_attentions.transpose(-2,-1)).flatten(1)) # B, N*N
        # out = self.co_attention(kge_heads, kge_tails)
        return F.softmax(out, -1)


    def GlosbalPool(self, h_data, t_data, h_data_edge, t_data_edge, i):
        # 节点注意力加权
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readouts[i](h_data.x, h_data.edge_index, edge_attr=h_data.edge_attr, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readouts[i](t_data.x, t_data.edge_index, edge_attr=t_data.edge_attr, batch=t_data.batch)
        # 节点readout
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch) # 节点全局池化, 每个分子图由一个embedding表示;global_add_pol:把所有节点特征表示相加
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)
        # 边readout
        h_data_edge.x = h_data.edge_attr
        t_data_edge.x = t_data.edge_attr
        h_global_graph_emb_edge = global_add_pool(h_data_edge.x, batch=h_data_edge.batch) # 边全局池化, B,dim
        t_global_graph_emb_edge = global_add_pool(t_data_edge.x, batch=t_data_edge.batch)
        h_global_graph_emb_edge = F.normalize(h_global_graph_emb_edge)
        t_global_graph_emb_edge = F.normalize(t_global_graph_emb_edge)
        # 融合图和节点表示
        # h_global_graph_emb = torch.concat([h_global_graph_emb, h_global_graph_emb_edge], -1)
        # t_global_graph_emb = torch.concat([t_global_graph_emb, t_global_graph_emb_edge], -1)
        h_global_graph_emb = h_global_graph_emb * h_global_graph_emb_edge
        t_global_graph_emb = t_global_graph_emb * t_global_graph_emb_edge
        h_global_graph_emb = self.to_trans[i](h_global_graph_emb)
        t_global_graph_emb = self.to_trans[i](t_global_graph_emb)
        h_global_graph_emb = F.normalize(h_global_graph_emb)
        t_global_graph_emb = F.normalize(t_global_graph_emb)
        return h_global_graph_emb, t_global_graph_emb


class CoAttentionLayer(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features
        self.w_q = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.w_k = nn.Parameter(torch.zeros(n_features, n_features//2))
        self.bias = nn.Parameter(torch.zeros(n_features // 2))
        self.a = nn.Parameter(torch.zeros(n_features//2))

        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.bias.view(*self.bias.shape, -1))
        nn.init.xavier_uniform_(self.a.view(*self.a.shape, -1))
    
    def forward(self, receiver, attendant):
        keys = receiver @ self.w_k
        queries = attendant @ self.w_q
        # values = receiver @ self.w_v
        values = receiver
        e_activations = queries.unsqueeze(-3) + keys.unsqueeze(-2) + self.bias
        e_scores = torch.tanh(e_activations) @ self.a
        # e_scores = e_activations @ self.a
        attentions = e_scores
        return attentions

# intra
class IntraAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, drug):
        x = drug.x
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1) # 按最后一维划分变成三份
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)
    
# inter
class InterAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, drug1, drug2):
        d1 = drug1.x
        d1 = self.norm(d1)
        d2 = drug2.x
        d2 = self.norm(d2)
        q = self.to_q(d1)
        kv = self.to_kv(d2).chunk(2, dim = -1) # 按最后一维划分变成两份
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)