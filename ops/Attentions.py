import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
import torch
from torch import nn


class KVGather(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, kv: Tensor, r_idx: Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
                               # (n, p^2, p^2, w^2, c_kv) without mem cpy
                               dim=2,
                               index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
                               # (n, p^2, k, w^2, c_kv)
                               )

        return topk_kv


class AttentionQKV(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, Q, K, V):
        # 计算注意力分数
        B, heads, N, C = Q.shape
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ V).transpose(1, 2).reshape(B, N, heads * C)

        return x


def combine_windows(windows, H, W):
    """
    Args:
        windows: (num_windows, B, window_size*window_size, C)
        H: Height of the original feature map
        W: Width of the original feature map
    Returns:
        x: (B, H, W, C)
    """
    num_windows, B, _, C = windows.shape
    window_size = int((windows.shape[2]) ** 0.5)

    windows = windows.view(num_windows, B, window_size, window_size, C)
    windows = windows.permute(1, 0, 2, 3, 4).contiguous()

    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

    return x

class Partition(nn.Module):
    def __init__(self,r):
        super().__init__()
        self.r = r

    def forward(self, x):  # 窗口划分
        """
        Args:
            x: (heads, H, W, C)
            window_size (int): window size
        Returns:
            windows: (heads,num_windows, window_size, window_size, C)
        """
        window_size = self.r
        B, heads, H, W, C = x.shape  # 特征图的形状分别代表，一次处理的样本数量，宽，高，通道
        x = x.view(B, heads, H // window_size, window_size, W // window_size, window_size, C)  # 窗口划分HW/MM,window_size=M

        # x.view()用于重新塑造张量 x 的形状而不改变其底层数据
        windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, heads, -1, window_size, window_size, C)
        # contiguous()：这是一个用于确保张量内存中元素连续排列的操作
        return windows


class WindowsSparseAttention(nn.Module):
    def __init__(self, scale, r, heads):
        super().__init__()
        self.r = r
        self.heads = heads
        self.scale = scale
        self.kvgather = KVGather()
        self.self_attention = AttentionQKV(scale)
        self.window_partition = Partition(r)
    def forward(self, q, k, v, indices):
        # indices.shape = (Batch, heads,num_windows,窗口号) 只对记录窗口编号的窗口注意力，因此为稀疏注意力
        window_size = self.r  # 每个区域大小为 4x4
        heads = self.heads
        group_size = window_size ** 2  # 每窗口包含的像素数量
        H = q.shape[2]
        # print(f'head{q.shape}')
        Q = self.window_partition(q)  # shape = (B,heads,num_windows,window_size,window_size,C)
        K = self.window_partition(k)  # shape = (B,heads,num_windows,window_size,window_size,C)
        V = self.window_partition(v)  # shape = (B,heads,num_windows,window_size,window_size,C)
        # print(f'head变成{Q.shape}')
        B, _, num_windows, window_size, window_size, C = Q.shape
        Q = Q.view(B, heads, num_windows, window_size * window_size, C)
        K = K.view(B, heads, num_windows, window_size * window_size, C)
        V = V.view(B, heads, num_windows, window_size * window_size, C)

        Q = Q.permute(2, 0, 1, 3, 4).reshape(num_windows * B, heads, window_size * window_size, C)

        # 创建一个空列表，用于存储切片后的张量

        # # 按照num_windows切片并将结果存入selected_Q
        # print(indices.shape)
        # 初始化选定的 K 和 V 列表
        # print(V.shape)
        # 遍历每个注意力头
        _, _, _, num_topk = indices.shape
        indices = indices.reshape(-1, num_windows, num_topk)

        K = self.kvgather(K.view(-1, num_windows, window_size * window_size, C), indices).view(B, heads, num_windows,
                                                                                               num_topk *
                                                                                               window_size * window_size,
                                                                                               C)
        V = self.kvgather(V.view(-1, num_windows, window_size * window_size, C), indices).view(B, heads, num_windows,
                                                                                               num_topk *
                                                                                               window_size * window_size,
                                                                                               C)
        # print(selected_V.shape)
        K = K.permute(2, 0, 1, 3, 4).reshape(num_windows * B, heads, -1, C)
        V = V.permute(2, 0, 1, 3, 4).reshape(num_windows * B, heads, -1, C)

        output = self.self_attention(Q, K, V)

        output = output.view(num_windows, B, -1, heads * C)
        # print(output.shape)
        output = combine_windows(output, H, H)

        return output
