import torch
from typing import Optional
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch_geometric.nn import TransformerConv, SAGPooling, LayerNorm, global_add_pool, Linear, GATv2Conv
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax

from torch_geometric.nn.inits import glorot, zeros
from einops import rearrange
import time


class GATEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int,
                 dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = nn.Parameter(torch.Tensor(1, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, in_channels))

        self.lin1 = Linear(in_channels , out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: torch.Tensor, edge_index: Adj) -> torch.Tensor:
        out = self.propagate(edge_index, x=x)
        out += self.bias
        return out

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor,
                index: torch.Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:

        x_j = F.leaky_relu_(self.lin1(x_j))
        alpha_j = (x_j * self.att_l).sum(dim=-1)
        alpha_i = (x_i * self.att_r).sum(dim=-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class Encode_Block(nn.Module):
    def __init__(self, in_features, n_heads, head_out_feats, edge_feature, dp):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats

        self.feature_conv = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        self.feature_conv2 = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        # self.feature_conv = GATv2Conv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        # self.feature_conv2 = GATv2Conv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        
        self.lin_up = Linear(64, 64, bias=True, weight_initializer='glorot')
        self.lin_up2 = Linear(64, 64, bias=True, weight_initializer='glorot')

        self.norm = LayerNorm(n_heads * head_out_feats)
        self.norm2 = LayerNorm(n_heads * head_out_feats)
    
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
    
    # shortcut
    # def forward(self, drug_data):

    #     # node_update
    #     x.x = self.feature_conv(drug_data.x, x.edge_index, x.edge_attr)
    #     x.x = F.elu(self.norm(x.x, x.batch))

    #     x.edge_attr = self.lin_up(x.edge_attr)
    #     x.edge_attr = F.normalize(x.edge_attr)
    #     x.edge_attr = F.elu(x.edge_attr)

    #     # global
    #     x.x = self.feature_conv2(x.x, x.edge_index, x.edge_attr)
    #     x.edge_attr = self.lin_up2(x.edge_attr)
    #     x.edge_attr = F.normalize(x.edge_attr)

    #     drug_data.x = drug_data.x + x.x
    #     drug_data.edge_attr = drug_data.edge_attr + x.edge_attr

    #     # node_shortcut
    #     drug_data.x = F.elu(drug_data.x)
    #     drug_data.edge_attr = F.elu(drug_data.edge_attr)

    #     return drug_data


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


class CoAttentionLayer3(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_dim = nn.Linear(dim, inner_dim, bias=False)
        self.W = nn.Parameter(torch.zeros(inner_dim, inner_dim))
        self.to_out = nn.Linear(inner_dim, 2)

        nn.init.xavier_uniform_(self.W)


    def forward(self, drug1, drug2):
        # durg: N,L,D
        drug1 = self.norm(drug1)
        drug2 = self.norm(drug2)
        
        d1 = self.to_dim(drug1)
        d2 = self.to_dim(drug2)
        
        d1 = rearrange(d1, 'b n (h d) -> b h n d', h = self.heads)
        d2 = rearrange(d2, 'b n (h d) -> b h n d', h = self.heads)

        att = torch.matmul(d1, d2.transpose(-1,-2)) * self.scale
        att = self.attend(att.flatten(-2))

        add = d1.unsqueeze(1) + d2.unsqueeze(2)
        out1 = torch.sum(att.unsqueeze(-1) * add.flatten(-3,-2), -2)
        
        out2 = rearrange(out1, 'b h d -> b (h d)', h = self.heads)

        return self.to_out(out2)


class MergeFD(nn.Module):
    def __init__(self, in_features_fp, in_features_desc, kge_dim):
        super().__init__()
        self.in_features_fp = in_features_fp
        self.in_features_desc = in_features_desc
        self.kge_dim = kge_dim
        self.reduction_fp = nn.Sequential(nn.Linear(self.in_features_fp, 512),
                                        #   nn.BatchNorm1d(4096),
                                          nn.ELU(),
                                          nn.Dropout(0.3),
                                          nn.Linear(512, self.kge_dim),
                                        #   nn.BatchNorm1d(1024),
                                          nn.ELU(),
                                          nn.Dropout(0.3)
                                          )
        
        self.reduction_desc = nn.Sequential(nn.Linear(self.in_features_desc, 256),
                                        #   nn.BatchNorm1d(256),
                                          nn.ELU(),
                                          nn.Dropout(0.3),
                                          nn.Linear(256, self.kge_dim),
                                        #   nn.BatchNorm1d(64),
                                          nn.ELU(),
                                          nn.Dropout(0.3))
        
        self.merge_fd = nn.Sequential(nn.Linear(self.kge_dim*2, self.kge_dim),
                                   nn.ELU())

    def forward(self,h_data_fin,h_data_desc,t_data_fin,t_data_desc):
        # 正则化
        h_data_fin = F.normalize(h_data_fin, 2, 1)
        h_data_desc = F.normalize(h_data_desc, 2, 1)
        
        t_data_fin = F.normalize(t_data_fin, 2 ,1)
        t_data_desc = F.normalize(t_data_desc, 2, 1)

        # 非线性变换
        h_data_fin = self.reduction_fp(h_data_fin)
        h_data_desc = self.reduction_desc(h_data_desc)

        t_data_fin = self.reduction_fp(t_data_fin)
        t_data_desc = self.reduction_desc(t_data_desc)

        h_fdmerge = torch.cat((h_data_fin, h_data_desc), dim=1) # 聚合
        h_fdmerge = F.normalize(h_fdmerge, 2, 1) # 正则化
        h_fdmerge = self.merge_fd(h_fdmerge) # 线性变换

        t_fdmerge = torch.cat((t_data_fin, t_data_desc), dim=1)
        t_fdmerge = F.normalize(t_fdmerge, 2, 1)
        t_fdmerge = self.merge_fd(t_fdmerge)

        return h_fdmerge, t_fdmerge, h_data_fin, h_data_desc, t_data_fin, t_data_desc
        # return h_fdmerge, t_fdmerge


class SD(nn.Module):
    def __init__(self, nclass, in_node_features, in_edge_features, hidd_dim, kge_dim, n_out_feats, n_heads, edge_feature, dp):
        super(SD, self).__init__()
        self.in_node_features = in_node_features
        self.in_edge_features = in_edge_features
        self.hidd_dim = hidd_dim
        self.kge_dim = kge_dim
        self.edge_feature = edge_feature
        self.n_blocks = len(n_heads)
        
        self.initial_node_feature = Linear(self.in_node_features, self.hidd_dim ,bias=True, weight_initializer='glorot')
        self.initial_edge_feature = Linear(self.in_edge_features, edge_feature ,bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(self.hidd_dim)
        
        self.encode_blocks = []
        for i, (head_out_feats, n_head) in enumerate(zip(n_out_feats, n_heads)):
            self.encode_blocks.append(Encode_Block(self.hidd_dim, n_head, head_out_feats, edge_feature, dp))

        self.encode_blocks = nn.ModuleList(self.encode_blocks)

        self.readout = SAGPooling(n_head * head_out_feats, min_score=-1) # min_score，注意力得分大于min_score节点被保留，设置-1表示所有节点做完自注意力后都被保留，或者ratio=1也保留全部节点
        # self.co_attention = CoAttentionLayer2(self.hidd_dim, 8, self.hidd_dim) 
        self.co_attention = CoAttentionLayer2(self.hidd_dim, 8, self.hidd_dim) 
        self.merge_all = nn.Sequential(nn.Linear(len(n_out_feats)**2, nclass))

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
            h_global_graph_emb, t_global_graph_emb = self.GlosbalPool(drug1, drug2, d1_edge, d2_edge) # 融合节点的全局特征表示
            
            repr_h.append(h_global_graph_emb)
            repr_t.append(t_global_graph_emb)

        repr_h = torch.stack((repr_h), dim=1) # B, layer num, D
        repr_t = torch.stack((repr_t), dim=1)
        
        head_attentions = self.co_attention(repr_h, repr_t) # 注意力，heads为key，tails为query
        tail_attentions = self.co_attention(repr_t, repr_h)

        out = self.merge_all(torch.matmul(head_attentions, tail_attentions.transpose(-2,-1)).flatten(1))
        
        # out = self.co_attention(kge_heads, kge_tails)
        return F.softmax(out, -1)


    def GlosbalPool(self, h_data, t_data, h_data_edge, t_data_edge):
        # 节点注意力加权
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readout(h_data.x, h_data.edge_index, edge_attr=h_data.edge_attr, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readout(t_data.x, t_data.edge_index, edge_attr=t_data.edge_attr, batch=t_data.batch)
        
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

        h_global_graph_emb = h_global_graph_emb * h_global_graph_emb_edge # 融合图和节点表示
        t_global_graph_emb = t_global_graph_emb * t_global_graph_emb_edge

        h_global_graph_emb = F.normalize(h_global_graph_emb)
        t_global_graph_emb = F.normalize(t_global_graph_emb)

        return h_global_graph_emb, t_global_graph_emb


class MVN_DDI(nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params, edge_feature,dp):
        super().__init__()
        self.in_node_features = in_node_features[0]
        self.in_node_features_fp = in_node_features[1]
        self.in_node_features_desc = in_node_features[2]
        self.in_edge_features = in_edge_features
        self.hidd_dim = hidd_dim
        self.kge_dim = kge_dim
        self.rel_total = rel_total
        self.n_blocks = len(blocks_params)
        
        self.initial_node_feature = Linear(self.in_node_features, self.hidd_dim ,bias=True, weight_initializer='glorot')
        self.initial_edge_feature = Linear(self.in_edge_features, edge_feature ,bias=True, weight_initializer='glorot')
        self.initial_node_norm = LayerNorm(self.hidd_dim)
        
        self.blocks = []
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = MVN_DDI_Block(self.hidd_dim, n_heads, head_out_feats, edge_feature, dp)
            # block = DeeperGCN(self.hidd_dim, n_heads, head_out_feats)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)

        self.co_attention = CoAttentionLayer(self.kge_dim) 
        self.KGE = RESCAL(self.rel_total, self.kge_dim)
        self.fdmer = MergeFD(self.in_node_features_fp, self.in_node_features_desc, self.kge_dim)
        # self.fdmer = MergeFD_trans(self.in_node_features_fp, self.in_node_features_desc, self.kge_dim, "Transformer")
        self.merge_all = nn.Sequential(nn.Linear(self.kge_dim * 2, self.kge_dim), nn.ReLU())

    def forward(self, triples):
        h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels, h_data_edge, t_data_edge = triples
        # h_data, h_data_fin, h_data_desc, t_data, t_data_fin, t_data_desc, rels = triples
        
        # 线性变换 55-64/128
        h_data.x = self.initial_node_feature(h_data.x) # 转换特征维数
        t_data.x = self.initial_node_feature(t_data.x)
        h_data.x = self.initial_node_norm(h_data.x, h_data.batch) # norm，正则化
        t_data.x = self.initial_node_norm(t_data.x, t_data.batch)
        h_data.x = F.elu(h_data.x) # 非线性激活，
        t_data.x = F.elu(t_data.x)

        h_data.edge_attr = self.initial_edge_feature(h_data.edge_attr)  # 边属性
        t_data.edge_attr = self.initial_edge_feature(t_data.edge_attr)
        h_data.edge_attr = F.elu(h_data.edge_attr)
        t_data.edge_attr = F.elu(t_data.edge_attr)

        
        # 4层gat 64-32*2
        repr_h = []
        repr_t = []
        for i, block in enumerate(self.blocks):
            out = block(h_data,t_data, h_data_edge, t_data_edge)
            # out = block(h_data,t_data,h_data_desc,t_data_desc)
            h_data = out[0]
            t_data = out[1]
            h_global_graph_emb = out[2]
            t_global_graph_emb = out[3]
            repr_h.append(h_global_graph_emb)
            repr_t.append(t_global_graph_emb)


        # fin-desc的融合模块，但融合模块返回的是前两个值，后面四个是在原值的基础上做的正则化和线性变换
        _, _, h_data_fin, h_data_desc, t_data_fin, t_data_desc = self.fdmer(h_data_fin,h_data_desc,t_data_fin,t_data_desc)

        repr_h_fd = []
        repr_t_fd = []
        for i in range(len(self.blocks)):
            repr_h_fd.append(F.normalize(repr_h[i] + h_data_fin)) # 把分子指纹融合到各层
            repr_t_fd.append(F.normalize(repr_t[i] + t_data_fin)) # 把分子指纹融合到各层
        
        repr_h = torch.stack((repr_h_fd), dim=1)
        repr_t = torch.stack((repr_t_fd), dim=1)
        kge_heads = repr_h #1024,4,128
        kge_tails = repr_t #1024,4,128
        attentions = self.co_attention(kge_heads, kge_tails) # 注意力，heads为key，tails为query
        scores = self.KGE(kge_heads, kge_tails, rels, attentions)

        return scores     


class MVN_DDI_Blck(nn.Module):
    def __init__(self, in_features, n_heads, head_out_feats, edge_feature, dp):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.feature_conv = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        self.lin_up = Linear(64, 64, bias=True, weight_initializer='glorot')
        
        self.feature_conv2 = TransformerConv(in_features, head_out_feats, n_heads, edge_dim=edge_feature,dropout=dp)
        self.lin_up2 = Linear(64, 64, bias=True, weight_initializer='glorot')
        
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)
        self.re_shape = Linear(64 + 128, 128, bias=True, weight_initializer='glorot')
        self.norm = LayerNorm(n_heads * head_out_feats)
        self.norm2 = LayerNorm(n_heads * head_out_feats)
        
        self.re_shape_e = Linear(64, 128, bias=True, weight_initializer='glorot')
    
    def forward(self, h_data,t_data,h_data_edge, t_data_edge):
        h_x_in = h_data.x
        t_x_in = t_data.x

        # node_update
        h_data, t_data = self.ne_update(h_data, t_data)

        # global
        h_data.x = self.feature_conv2(h_data.x, h_data.edge_index, h_data.edge_attr)
        t_data.x = self.feature_conv2(t_data.x, t_data.edge_index, t_data.edge_attr)
        h_data.edge_attr = self.lin_up2(h_data.edge_attr)
        t_data.edge_attr = self.lin_up2(t_data.edge_attr)

        h_global_graph_emb, t_global_graph_emb = self.Globa4lPool(h_data, t_data, h_data_edge, t_data_edge) # 全局池化，融合节点属性和边属性

        # node_shortcut
        h_data.x = F.elu(self.norm2(h_data.x, h_data.batch)) # 断链接
        t_data.x = F.elu(self.norm2(t_data.x, t_data.batch))
        
        h_data.edge_attr = F.elu(h_data.edge_attr)
        t_data.edge_attr = F.elu(t_data.edge_attr)

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb

    def ne_update(self, h_data, t_data):
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index, h_data.edge_attr)
        t_data.x = self.feature_conv(t_data.x, t_data.edge_index, t_data.edge_attr)
        h_data.x = F.elu(self.norm(h_data.x, h_data.batch))
        t_data.x = F.elu(self.norm(t_data.x, t_data.batch))

        h_data.edge_attr = self.lin_up(h_data.edge_attr)
        t_data.edge_attr = self.lin_up(t_data.edge_attr)
        h_data.edge_attr = F.elu(h_data.edge_attr)
        t_data.edge_attr = F.elu(t_data.edge_attr)

        return h_data, t_data

    def GlobalPool(self, h_data, t_data, h_data_edge, t_data_edge):
        h_att_x, att_edge_index, att_edge_attr, h_att_batch, att_perm, h_att_scores= self.readout(h_data.x, h_data.edge_index, edge_attr=h_data.edge_attr, batch=h_data.batch)
        t_att_x, att_edge_index, att_edge_attr, t_att_batch, att_perm, t_att_scores= self.readout(t_data.x, t_data.edge_index, edge_attr=t_data.edge_attr, batch=t_data.batch)
        h_global_graph_emb = global_add_pool(h_att_x, h_att_batch) # 节点全局池化
        t_global_graph_emb = global_add_pool(t_att_x, t_att_batch)

        h_data_edge.x = h_data.edge_attr
        t_data_edge.x = t_data.edge_attr
        h_global_graph_emb_edge = global_add_pool(h_data_edge.x, batch=h_data_edge.batch) # 边全局池化
        t_global_graph_emb_edge = global_add_pool(t_data_edge.x, batch=t_data_edge.batch)
        h_global_graph_emb_edge = F.elu(self.re_shape_e(h_global_graph_emb_edge))
        t_global_graph_emb_edge = F.elu(self.re_shape_e(t_global_graph_emb_edge))

        h_global_graph_emb = h_global_graph_emb * h_global_graph_emb_edge
        t_global_graph_emb = t_global_graph_emb * t_global_graph_emb_edge
        return h_global_graph_emb, t_global_graph_emb
