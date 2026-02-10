import torch
import torch.nn.functional as F

# def get_cl_loss(x1, x2, T=0.1):
#     batch_size, _ = x1.size()
#     x1_abs = x1.norm(dim=1)
#     x2_abs = x2.norm(dim=1)
#     sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
#     sim_matrix = torch.exp(sim_matrix / T)
#     pos_sim = sim_matrix[range(batch_size), range(batch_size)]
#     loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
#     loss = - torch.log(loss1).mean()
#     return loss


def get_cl_loss(x1, x2, T=0.5):
    # 归一化输入特征（关键步骤，避免除零错误）
    x1 = F.normalize(x1, dim=1)
    x2 = F.normalize(x2, dim=1)

    batch_size = x1.size(0)

    # 计算相似度矩阵（余弦相似度）
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / T

    # 数值稳定性处理：减去每行最大值
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True).values

    # 计算指数并取对数（InfoNCE核心）
    sim_matrix_exp = torch.exp(sim_matrix)
    pos_sim = sim_matrix_exp[range(batch_size), range(batch_size)]
    denominator = sim_matrix_exp.sum(dim=1)

    # 计算最终损失（避免分母为零）
    loss = -torch.log(pos_sim / denominator).mean()

    return loss


def get_total_loss(label_loss, graph_data, smiles_data, alpha=0.2):
    loss1 = label_loss
    loss2 = get_cl_loss(graph_data, smiles_data)

    return loss1 + alpha * loss2, loss1, loss2