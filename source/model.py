# -*- coding: utf-8 -*-
"""
@Author  : Weimin Zhu
@Time    : 2021-09-28
@File    : model.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.nn import Linear, Sequential, Parameter, Bilinear
from torch_geometric.utils import to_dense_batch

from torch_scatter import scatter
from torch_geometric.nn import global_add_pool, GATConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.nn.pool.pool import pool_batch
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from transformers import RobertaPreTrainedModel, RobertaModel


# ---------------------------------------
# Attention layers
# ---------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.mlp = Sequential(
            Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            Linear(channels // reduction, channels, bias=False),
        )

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.mlp)

    def forward(self, x, batch, size=None):
        max_result = scatter(x, batch, dim=0, dim_size=size, reduce='max')
        sum_result = scatter(x, batch, dim=0, dim_size=size, reduce='sum')
        max_out = self.mlp(max_result)
        sum_out = self.mlp(sum_result)
        y = torch.sigmoid(max_out + sum_out)
        y = y[batch]
        return x * y


# ---------------------------------------
# Neural tensor networks conv
# ---------------------------------------
class NTNConv(MessagePassing):

    def __init__(self, in_channels, out_channels, slices, dropout, edge_dim=None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(NTNConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.slices = slices
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.weight_node = Parameter(torch.Tensor(in_channels,
                                                  out_channels))
        if edge_dim is not None:
            self.weight_edge = Parameter(torch.Tensor(edge_dim,
                                                      out_channels))
        else:
            self.weight_edge = self.register_parameter('weight_edge', None)

        self.bilinear = Bilinear(out_channels, out_channels, slices, bias=False)

        if self.edge_dim is not None:
            self.linear = Linear(3 * out_channels, slices)
        else:
            self.linear = Linear(2 * out_channels, slices)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_node)
        glorot(self.weight_edge)
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):

        x = torch.matmul(x, self.weight_node)

        if self.weight_edge is not None:
            assert edge_attr is not None
            edge_attr = torch.matmul(edge_attr, self.weight_edge)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_attr):
        score = self.bilinear(x_i, x_j)
        if edge_attr is not None:
            vec = torch.cat((x_i, edge_attr, x_j), 1)
            block_score = self.linear(vec)  # bias already included
        else:
            vec = torch.cat((x_i, x_j), 1)
            block_score = self.linear(vec)
        scores = score + block_score
        alpha = torch.tanh(scores)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        dim_split = self.out_channels // self.slices
        out = torch.max(x_j, edge_attr).view(-1, self.slices, dim_split)

        out = out * alpha.view(-1, self.slices, 1)
        out = out.view(-1, self.out_channels)
        return out

    def __repr__(self):
        return '{}({}, {}, slices={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.slices)


# ---------------------------------------
# HiGNN backbone
# ---------------------------------------
def build_model_HiGNN(cfg):
    model = HiGNN(
          # GNN
          gnn_in_channels=46,
          gnn_hidden_channels=cfg.MODEL.GNN.HID_DIM,
          gnn_out_channels=cfg.MODEL.GNN.OUT_DIM,
          gnn_edge_dim=10,
          gnn_num_layers=cfg.MODEL.GNN.DEPTH,
          gnn_dropout=cfg.MODEL.GNN.DROPOUT,
          gnn_slices=cfg.MODEL.GNN.SLICES,
          gnn_f_att=cfg.MODEL.GNN.F_ATT,
          r=cfg.MODEL.GNN.R,
          gnn_brics=cfg.MODEL.GNN.BRICS,
          gnn_cl=cfg.LOSS.CL_LOSS,
          # SEQ
          seq_out_dim=cfg.MODEL.SEQ.OUT_DIM,
          # Attention
          att_num_heads=cfg.MODEL.ATT.NUM_HEADS,
          att_dropout=cfg.MODEL.ATT.DROPOUT
    )

    return model


class HiGNN(torch.nn.Module):
    """Hierarchical informative graph neural network for molecular representation."""


    def __init__(self,
                 # GNN
                 gnn_in_channels, gnn_hidden_channels, gnn_out_channels, gnn_edge_dim, gnn_num_layers,
                 gnn_slices, gnn_dropout, gnn_f_att, r, gnn_brics, gnn_cl,
                 # SEQ
                 seq_out_dim,
                 # Attention
                 att_num_heads,
                 att_dropout,
                 ):
        super(HiGNN, self).__init__()

        self.gnn_hidden_channels = gnn_hidden_channels
        self.gnn_num_layers = gnn_num_layers
        self.gnn_dropout = gnn_dropout

        self.gnn_f_att = gnn_f_att
        self.gnn_brics = gnn_brics
        self.gnn_cl = gnn_cl

        # atom feature transformation
        self.lin_a = Linear(gnn_in_channels, gnn_hidden_channels)
        self.lin_b = Linear(gnn_edge_dim, gnn_hidden_channels)

        # 对齐graph和smiles
        self.align_graph_to_smiles = Linear(gnn_hidden_channels * 2, seq_out_dim)
        # 对齐官能团
        self.align_functional = Linear(gnn_hidden_channels, seq_out_dim)
        # self.align_functional = Linear(gnn_hidden_channels, seq_out_dim * 2)
        '''
                # 跨膜态注意力机制
        self.cross_modal_attention_layer = CrossModalAttentionLayer(embed_dim=seq_out_dim, num_heads=att_num_heads, dropout=att_dropout)
        '''
        # self.cross_modal_attention_layer = CrossModalAttentionLayer(embed_dim=seq_out_dim, num_heads=att_num_heads,
        #                                                             dropout=att_dropout)

        # self.gate_fusion =CrossAttentionFusion(
        #                         hidden_dim=768,
        #                         num_heads=4,
        #                         dropout=0.1,
        #                         use_gate=True
        #                     )
        # self.gate_fusion = EnhancedFusion(
        #                         hidden_dim=768,
        #                         num_heads=4,
        #                         num_layers=2,
        #                         dropout=0.1,
        #                         drop_path=0.1
        #                     )
        self.gate_fusion =CrossAttentionFusion(768)



        # convs block
        self.atom_convs = torch.nn.ModuleList()
        for _ in range(gnn_num_layers):
            conv = NTNConv(gnn_hidden_channels, gnn_hidden_channels, slices=gnn_slices,
                           dropout=gnn_dropout, edge_dim=gnn_hidden_channels)
            self.atom_convs.append(conv)

        self.lin_gate = Linear(3 * gnn_hidden_channels, gnn_hidden_channels)

        if self.gnn_f_att:
            self.feature_att = FeatureAttention(channels=gnn_hidden_channels, reduction=r)

        if self.gnn_brics:
            # mol-fra attention
            self.cross_att = GATConv(gnn_hidden_channels, gnn_hidden_channels, heads=4,
                                     dropout=gnn_dropout, add_self_loops=False,
                                     negative_slope=0.01, concat=False)
            # self.functional_att = GATv2Conv(in_channels=gnn_hidden_channels,
            #                            out_channels=gnn_hidden_channels,
            #                            heads=4,
            #                            dropout=gnn_dropout,
            #                            add_self_loops=False,
            #                            negative_slope=0.01,
            #                            concat=False)
            self.functional_motif = MotifEnhancer(in_channels=gnn_hidden_channels,
                                              out_channels=gnn_hidden_channels,
                                              gnn_dropout=gnn_dropout)



        if self.gnn_brics:
            ''' Cross Modal Attn '''
            # self.out = Linear(seq_out_dim*2, gnn_out_channels)
            self.out = Linear(seq_out_dim, gnn_out_channels)
        else:
            self.out = Linear(gnn_hidden_channels, gnn_out_channels)

        if self.gnn_cl:
            self.lin_project = Linear(gnn_hidden_channels, int(gnn_hidden_channels/2))

        self.reset_parameters()

    def reset_parameters(self):

        self.lin_a.reset_parameters()
        self.lin_b.reset_parameters()

        for conv in self.atom_convs:
            conv.reset_parameters()

        self.lin_gate.reset_parameters()

        if self.gnn_f_att:
            self.feature_att.reset_parameters()

        if self.gnn_brics:
            self.cross_att.reset_parameters()
            # self.functional_att.reset_parameters()

        self.out.reset_parameters()

        if self.gnn_cl:
            self.lin_project.reset_parameters()




    '''
    No fine-tuning Bert (Freeze) + HiGNN
    '''
    def forward(self, data, xgb_flag):
        # get mol input
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        x = F.relu(self.lin_a(x))  # (N, 46) -> (N, hidden_channels)
        edge_attr = F.relu(self.lin_b(edge_attr))  # (N, 10) -> (N, hidden_channels)

        # mol conv block
        for i in range(0, self.gnn_num_layers):
            h = F.relu(self.atom_convs[i](x, edge_index, edge_attr))
            beta = self.lin_gate(torch.cat([x, h, x - h], 1)).sigmoid()
            x = beta * x + (1 - beta) * h
            if self.gnn_f_att:
                x = self.feature_att(x, batch)

        mol_vec = global_add_pool(x, batch).relu_()

        if self.gnn_brics:
            # get fragment input
            fra_x = data.x
            fra_edge_index = data.fra_edge_index
            fra_edge_attr = data.fra_edge_attr
            cluster = data.cluster_index

            fra_x = F.relu(self.lin_a(fra_x))  # (N, 46) -> (N, hidden_channels)
            fra_edge_attr = F.relu(self.lin_b(fra_edge_attr))  # (N, 10) -> (N, hidden_channels)

            # fragment convs block
            for i in range(0, self.gnn_num_layers):
                fra_h = F.relu(self.atom_convs[i](fra_x, fra_edge_index, fra_edge_attr))
                beta = self.lin_gate(torch.cat([fra_x, fra_h, fra_x - fra_h], 1)).sigmoid()
                fra_x = beta * fra_x + (1 - beta) * fra_h
                if self.gnn_f_att:
                    fra_x = self.feature_att(fra_x, cluster)

            fra_x = global_add_pool(fra_x, cluster).relu_()

            # get fragment batch
            cluster, perm = consecutive_cluster(cluster)
            fra_batch = pool_batch(perm, data.batch)

            # molecule-fragment attention
            row = torch.arange(fra_batch.size(0), device=batch.device)
            mol_fra_index = torch.stack([row, fra_batch], dim=0)
            fra_vec = self.cross_att((fra_x, mol_vec), mol_fra_index).relu_()
            motif_vec = self.functional_motif(fra_x, mol_vec, mol_fra_index)  # 官能团注意力


            vectors_concat = list()
            vectors_concat.append(mol_vec)
            vectors_concat.append(fra_vec)

            out = torch.cat(vectors_concat, 1)

            graph_data = self.align_graph_to_smiles(out)
            # graph_data = out

            smiles_data = torch.squeeze(torch.tensor(data.smiles_vec), dim=1).cuda()

            # functional_vec = self.align_functional(motif_vec)
            functional_vec = self.align_functional(motif_vec)

            # 融合三种模态的特征
            out, gate_weight = self.gate_fusion(graph_data, smiles_data)
            # out = torch.cat([graph_data, smiles_data], dim=1)
            out = out + functional_vec
            # out = self.multi_modal_transformer(combined_input)

            if xgb_flag == True:
                return out, data.smiles, gate_weight
            else:
                if self.gnn_cl:
                    out = F.dropout(out, p=self.gnn_dropout, training=self.training)
                    return self.out(out), graph_data, smiles_data, self.lin_project(mol_vec).relu_(), self.lin_project(fra_vec).relu_()

                out = F.dropout(out, p=self.gnn_dropout, training=self.training)
                return self.out(out), graph_data, smiles_data,None,None
                # return out




class CrossAttentionFusion(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_heads=4,
            dropout=0.1,
            use_gate=True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gate = use_gate

        # 交叉注意力层（双向：图特征 <-> SMILES特征）
        self.cross_attn_graph_to_smiles = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_smiles_to_graph = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 层归一化与残差连接
        self.norm_graph = nn.LayerNorm(hidden_dim)
        self.norm_smiles = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 门控机制（可选）
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Sigmoid()
            )

    def forward(self, graph_embedding, smiles_embedding):
        """
        Args:
            graph_embedding: 图特征 [batch_size, hidden_dim]
            smiles_embedding: SMILES 特征 [batch_size, hidden_dim]
        Returns:
            fused_feature: 融合后的特征 [batch_size, hidden_dim]
        """

        # 确保输入维度一致
        assert graph_embedding.shape == smiles_embedding.shape
        gate_weight = None

        # 双向交叉注意力
        # 1. 图特征作为 Query，SMILES 特征作为 Key/Value
        graph_query = graph_embedding.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        attn_graph_to_smiles, _ = self.cross_attn_graph_to_smiles(
            query=graph_query,
            key=smiles_embedding.unsqueeze(1),
            value=smiles_embedding.unsqueeze(1),
            need_weights=False
        )
        attn_graph_to_smiles = self.dropout(attn_graph_to_smiles.squeeze(1))

        # 2. SMILES 特征作为 Query，图特征作为 Key/Value
        smiles_query = smiles_embedding.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        attn_smiles_to_graph, _ = self.cross_attn_smiles_to_graph(
            query=smiles_query,
            key=graph_embedding.unsqueeze(1),
            value=graph_embedding.unsqueeze(1),
            need_weights=False
        )
        attn_smiles_to_graph = self.dropout(attn_smiles_to_graph.squeeze(1))

        # 残差连接 + 层归一化
        fused_graph = self.norm_graph(graph_embedding + attn_graph_to_smiles)
        fused_smiles = self.norm_smiles(smiles_embedding + attn_smiles_to_graph)

        # 特征融合（拼接后投影或门控）
        if self.use_gate:
            # 门控机制：动态调整两个模态的贡献
            gate_input = torch.cat([fused_graph, fused_smiles], dim=1)
            gate_weight = self.gate(gate_input)  # [batch_size, hidden_dim]
            fused_feature = gate_weight * fused_graph + (1 - gate_weight) * fused_smiles
        else:
            # 简单相加
            fused_feature = fused_graph + fused_smiles

        return fused_feature, gate_weight



class MotifEnhancer(nn.Module):
    def __init__(self, in_channels, out_channels, gnn_dropout):
        super(MotifEnhancer, self).__init__()
        self.gat =  GATConv(in_channels, out_channels, heads=4,
                                     dropout=gnn_dropout, add_self_loops=False,
                                     negative_slope=0.01, concat=False)
        # 添加批量归一化层
        self.bn = nn.BatchNorm1d(out_channels)
        #  激活函数
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)


    def forward(self, fra_vec, mol_vec, mol_fra_index):
        fra_vec = self.gat((fra_vec, mol_vec), mol_fra_index)
        functionnal_vec = self.leaky_relu(fra_vec)
        return functionnal_vec



