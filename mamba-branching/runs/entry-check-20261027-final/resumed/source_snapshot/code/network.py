from operator import is_
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import functools
from mamba_ssm import Mamba2, Mamba

class TreeNet(nn.Module):
    def __init__(self, input_size=61, hidden_size=[128, 256, 512, 1024], output_size=896):
        super(TreeNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(hidden_size)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size[i]))
            else:
                self.layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.layers.append(nn.Linear(hidden_size[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))    
        x = self.layers[-1](x)
        

        return x
        

class VariableNet(nn.Module):
    def __init__(self, input_size=25, hidden_size=[128, 256, 512], output_size=896):
        super(VariableNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(hidden_size)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size[i]))
            else:
                self.layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.layers.append(nn.Linear(hidden_size[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

class ProjectionNet(nn.Module):
    def __init__(
            self, tree_input_size=61, tree_hidden_size=[128, 256, 512, 1024],
            var_input_size=25, var_hidden_size=[128, 256, 512, 1024],
            output_size=896
        ):
        super(ProjectionNet, self).__init__()

        self.tree_net = TreeNet(input_size=tree_input_size, hidden_size=tree_hidden_size, output_size=output_size)
        # self.tree_net = TreeGateNet(infimum=output_size)
        self.var_net = VariableNet(input_size=var_input_size, hidden_size=var_hidden_size, output_size=output_size)

        self.temperature = 0.07
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))

    def forward(self, tree_x, var_x, all_vars):

        # tree_x.shape = (32, 61)
        # var_x.shape = (32, 25)
        # all_vars.shape = (32, max_candidats, 25)
        # tree_x = self.tree_net(all_vars, tree_x)  # shape=(32, 896)
        tree_x = self.tree_net(tree_x)  # shape=(32, 896)
        var_x = self.var_net(var_x)     # shape=(32, 896)

        all_vars = self.var_net(all_vars)  # shape=(32, max_candidats, 896)

        # l2归一化
        tree_x = F.normalize(tree_x, p=2, dim=-1)
        var_x = F.normalize(var_x, p=2, dim=-1)
        all_vars = F.normalize(all_vars, p=2, dim=-1)

        # 计算可学习的temperature
        logit_scale = self.logit_scale.exp()  # 确保scale为正数
        
        # 缩放相似度矩阵
        logits_per_image = logit_scale * (tree_x @ var_x.t())
        logits_per_text = logits_per_image.t()
        
        # 创建标签（对角线为正样本）
        batch_size = var_x.shape[0]
        labels = torch.arange(batch_size, device=var_x.device)
        
        # 双向对比损失
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        
        # 对称损失
        loss = (loss_i + loss_t) / 2

        return loss, tree_x, var_x, all_vars

class OutputNet(nn.Module):
    def __init__(self, input_size=896, output_size = 25, hidden_size=[512, 256, 128]):
        super(OutputNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(hidden_size)):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size[i]))
            else:
                self.layers.append(nn.Linear(hidden_size[i-1], hidden_size[i]))
        self.layers.append(nn.Linear(hidden_size[-1], output_size))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

def get_norm_layer(norm_type='none'):
    """
    :param norm_type: str, the name of the normalization layer: batch | instance | layer | none
    :return:
        norm_layer, a normalization layer
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm1d, affine=False, track_running_stats=False)
    elif norm_type == 'layer':
        norm_layer = functools.partial(nn.LayerNorm)
    elif norm_type == 'none':
        norm_layer = functools.partial(nn.Identity)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# TreeGate
class TreeGateBranchingNet(nn.Module):
    """
    TreeGate specific network.
    """
    def __init__(self, branch_size, tree_state_size, dim_reduce_factor, infimum=8, norm='none', depth=2,
                 hidden_size=128):
        super(TreeGateBranchingNet, self).__init__()
        norm_layer = get_norm_layer(norm)
        self.norm = norm
        self.branch_size = branch_size              # 看输入，是和hidden_size一致的
        self.tree_state_size = tree_state_size      # node_dim + mip_dim

        # branch_size = 64 = hidden_size
        # infimum = 1024
        # dim_reduce_factor = 2

        # 算出n_layers
        self.dim_reduce_factor = dim_reduce_factor  
        self.infimum = infimum
        self.n_layers = 0
        self.depth = depth
        self.hidden_size = hidden_size
        
        # 从branch_size开始，乘以dim_reduce_factor，直到大于infimum
        if infimum > branch_size:
            unit_count = branch_size
            while unit_count <= infimum:
                unit_count *= dim_reduce_factor
                self.n_layers += 1
        else:
            unit_count = infimum
            while unit_count < branch_size:
                unit_count *= dim_reduce_factor
                self.n_layers += 1
            
        # 同样，先构造一个bracnhing net, 和no tree一样
        # 加了个n_units_dict，用来计算H = h + h*2 + h*4 +... + infimum，并且存储每个layer对应的输出维度
        self.n_units_dict = dict.fromkeys(range(self.n_layers))
        self.BranchingNet = nn.ModuleList()
        input_dim = hidden_size
        for i in range(self.n_layers):
            if infimum > branch_size:
                output_dim = int(input_dim * dim_reduce_factor)
            else:
                output_dim = int(input_dim / dim_reduce_factor)
                
            self.n_units_dict[i] = input_dim
            if i < self.n_layers - 1:
                layer = [nn.Linear(input_dim, output_dim),
                         norm_layer(output_dim),
                         nn.ReLU(True)]
            elif i == self.n_layers - 1:
                layer = [nn.Linear(input_dim, infimum)]  # Dense output
            input_dim = output_dim
            self.BranchingNet.append(nn.Sequential(*layer))

        # define the GatingNet
        # 定义gating net
        self.GatingNet = []

        # n_attentional_units = H = h + h*2 + h*4 +... + infimum
        self.n_attentional_units = sum(self.n_units_dict.values())
        
        # Tree输入（61维），一次到达H
        if depth == 1:
            self.GatingNet += [nn.Linear(tree_state_size, self.n_attentional_units),
                               nn.Sigmoid()]
        
        # 否则，先到hidden_size，然后过depth-2次hidden_size到hiddent_size，最后再到H
        else:
            self.GatingNet += [nn.Linear(tree_state_size, hidden_size),
                               nn.ReLU(True)]
            for i in range(depth - 2):
                self.GatingNet += [nn.Linear(hidden_size, hidden_size),
                                   nn.ReLU(True)]
            self.GatingNet += [nn.Linear(hidden_size, self.n_attentional_units),
                               nn.Sigmoid()]
            self.GatingNet = nn.Sequential(*self.GatingNet)

    def forward(self, cands_state_mat, tree_state, is_multi = True):
        # tree, 拼接node和mip

        # gate输出attn_weights，维度为H = h + h/2 + h/4 +... + infimum
        attn_weights = self.GatingNet(tree_state)
        start_slice_idx = 0
        for index, layer in enumerate(self.BranchingNet):
            end_slice_idx = start_slice_idx + self.n_units_dict[index]  # 0~h-1, h ~ h+h/2 - 1, ...
            
            # clip训练中
            if len(cands_state_mat.shape) == 3:
                attn_slice = attn_weights[:,start_slice_idx:end_slice_idx]
            
            # llm训练中，len(cands_state_mat.shape) == 4
            else:
                # 选出当前层，对应的attn_weights，作为gate
                attn_slice = attn_weights[:,:,start_slice_idx:end_slice_idx]
            
            attn_slice = attn_slice.unsqueeze(-2)  # 逐步扩展
            cands_state_mat = cands_state_mat * attn_slice  # No in-place operations, bad for .backward()
            
            # 调制之后，再过branching net
            cands_state_mat = layer(cands_state_mat)
            
            start_slice_idx = end_slice_idx
        
        # branch net输出(candidate_num, infimum)，做一个平均
        # 不做平均池化，做max池化；shape = (batch_size, infimum)
        # 如果是llm，则shape = (batch_size, seq_len, infimum)
        # cands_state_mat.shape = (batch_size, candidate_num, infimum)
        # 或 (batch_size, seq_len, candidate_num, infimum)
        
        # cands_prob = cands_state_mat.max(dim=-2)[0]  # Keep the axis
        cands_prob = cands_state_mat.mean(dim=-1)      # shape = (batch_size, candidate_num) 

        if is_multi:
            return cands_state_mat, cands_prob
        else:
            return cands_prob

class TreeGateNet(nn.Module):
    """
    TreeGate policy.
    """
    def __init__(self, var_dim=25, node_dim=8, mip_dim=53, hidden_size=64, depth=3, dropout=0.0, dim_reduce_factor=2, infimum=768,
                 norm='none'):
        """
        :param var_dim: int, dimension of variable state
        :param node_dim: int, dimension of node state
        :param mip_dim: int, dimension of mip state
        :param hidden_size: int, hidden size parameter for the network
        :param depth: int, depth parameter for the network
        :param dropout: float, dropout parameter for the network
        :param dim_reduce_factor: int, Dimension reduce factor of the network
        :param infimum: int, infimum parameter of the network
        :param norm: str, normalization type of the network
        """
        super(TreeGateNet, self).__init__()
        self.dropout = dropout
        self.norm = norm

        # define the dimensionality of the features and the hidden states
        self.var_dim = var_dim
        self.node_dim = node_dim
        self.mip_dim = mip_dim
        self.hidden_size = hidden_size
        self.depth = depth

        # define CandidateEmbeddingNet
        self.CandidateEmbeddingNet = [nn.Linear(var_dim, hidden_size)]
        self.CandidateEmbeddingNet = nn.Sequential(*self.CandidateEmbeddingNet)

        # define the TreeGateBranchingNet
        self.TreeGateBranchingNet = TreeGateBranchingNet(hidden_size, node_dim + mip_dim, dim_reduce_factor,
                                                         infimum, norm, depth, hidden_size)

        # self.output_layer = nn.Sequential(
        #     nn.Linear(infimum, int(infimum/2)),
        #     nn.ReLU(True),
        #     nn.Linear(int(infimum/2), int(infimum/4)),
        #     nn.ReLU(True),
        #     nn.Linear(int(infimum/4), 1),
        # )

        
        # do the Xavier initialization for the linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(tensor=m.weight, gain=nn.init.calculate_gain('relu'))

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cands_state_mat, state, is_multi = True):
        # go through the CandidateEmbeddingNet
        cands_state_mat = self.CandidateEmbeddingNet(cands_state_mat)

        # go through the TreeGateBranchingNet
        res = self.TreeGateBranchingNet(cands_state_mat, state, is_multi)

        # if is_multi:
        #     cands_prob, cands_state_mat = res
        #     # 过一个输出层
        #     # logits = self.output_layer(cands_state_mat).squeeze(-1)  # shape = (b, max_cands_num)
        #     return cands_prob, cands_state_mat
        # else:
        #     return res # cands_probs
        
        return res # is_multi: cands_state_mat, cands_probs; else: cands_probs


# model = Mamba2(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=8, # Model dimension d_model
#     d_state=64,  # SSM state expansion factor, typically 64 or 128
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to(args.device)


# class GroupedMamba(nn.Module):
#     def __init__(self, d_model, max_seq_length):
#         super().__init__()
#         self.mamba = Mamba(
#             d_model=8, # Model dimension d_model
#             d_state=64,  # SSM state expansion factor, typically 64 or 128
#             d_conv=4,    # Local convolution width
#             expand=2,    # Block expansion factor
#         )
#         # 定义组间位置编码（可学习或固定）
#         self.position_embed = nn.Embedding(max_seq_length, d_model)  # 与输入同维度        
#     def forward(self, x, position_ids):
#         position_embed = self.position_embed(position_ids)
#         x = x + position_embed  # 将位置编码加到输入
#         return self.mamba(x)

class AttentionReducer(nn.Module):
    def __init__(self, feature_dim=8):
        super(AttentionReducer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, max_seq_len, candidate_num, infium)
        if len(x.shape) == 4:
            batch_size, max_seq_len, candidate_num, infium = x.shape
            # 计算注意力权重: (batch_size, max_seq_len, candidate_num, 1)
            attn_weights = self.attention(x)
            
            # 加权求和: (batch_size, max_seq_len, 8)
            output = torch.sum(x * attn_weights, dim=2)
            
            # 为了满足后面的结构，特意改一下把第2个维度留出来
            output = output.view(batch_size, max_seq_len, 1, infium)
            
            return output
        elif len(x.shape) == 3:
            batch_size, candidate_num, infium = x.shape
            # 计算注意力权重: (batch_size, candidate_num, 1)
            attn_weights = self.attention(x)
            
            # 加权求和: (batch_size, 8)
            output = torch.sum(x * attn_weights, dim=1)
            
            output = output.view(batch_size, 1, infium)

            return output
    
class GroupedMamba(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model, # Model dimension d_model
            d_state=64,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        # 定义组间位置编码（可学习或固定）
        self.position_embed = nn.Embedding(max_seq_length, d_model)  # 与输入同维度        
        # self.projection = nn.Sequential(
        #     nn.Linear(d_model, d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(d_model // 2, 1),
        # )
        # self.reducer = AttentionReducer(feature_dim=d_model)
    def forward(self, x, position_ids):
        position_embed = self.position_embed(position_ids)
        x = x + position_embed  # 将位置编码加到输入
        return self.mamba(x)
    
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, max_positions=512, num_layers=6, nhead=8, dim_feedforward=32, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.embed_size = embed_size
        
        # 可学习的位置嵌入（根据位置ID查找）
        self.position_embeddings = nn.Embedding(max_positions, embed_size)
        
        # 使用PyTorch内置的TransformerDecoderLayer和TransformerDecoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 使用(batch, seq, feature)格式
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 用于生成目标序列的mask
        self.register_buffer("tgt_mask", None)
        
    def _generate_square_subsequent_mask(self, sz):
        """生成用于自回归解码的mask"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, x, position_ids):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, embed_size)
            pos_ids: 位置ID张量，形状为 (batch_size, seq_len)
        Returns:
            输出张量，形状与输入相同 (batch_size, seq_len, embed_size)
        """
        # 获取位置嵌入 (batch_size, seq_len, embed_size)
        pos_embeddings = self.position_embeddings(position_ids)
        
        # 添加位置嵌入到输入中
        x = x + pos_embeddings
        
        # 生成目标mask（用于自回归）
        if self.tgt_mask is None or self.tgt_mask.size(0) != x.size(1):
            self.tgt_mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # 由于我们只是做decoder-only的转换，没有encoder的输出，所以memory设为x
        output = self.transformer_decoder(
            tgt=x,
            memory=x,
            tgt_mask=self.tgt_mask,
            memory_mask=None
        )
        
        return output