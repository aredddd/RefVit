import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange
from ops import Attentions


class ReshapeLinear(nn.Module):
    def __init__(self, in_features, out_features, map_k=3):
        super(ReshapeLinear, self).__init__()
        assert map_k <= in_features
        self.in_features = in_features
        self.out_features = out_features
        # 将 linear 的权重设置为缓冲区
        self.register_buffer('weight', torch.zeros(out_features, in_features))
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.convmap = nn.Conv1d(in_channels=in_features, out_channels=out_features,
                                 kernel_size=map_k, stride=1, padding=map_k // 2, bias=False)
        nn.init.xavier_uniform_(self.convmap.weight)  # 将 convmap 设置为可学习的参数并初始化

    def forward(self, inputs):
        # Reshape the original weight using convmap
        origin_weight = self.weight.view(1, self.out_features, self.in_features)
        weight = self.weight + self.convmap(origin_weight).view(self.out_features, self.in_features)
        # Perform linear operation
        return F.linear(inputs, weight, bias=self.linear.bias)


class MultiStreamAttention(nn.Module):
    """
    attention
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=4,topk = 4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.topk = topk
        self.r = r
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.downsample = nn.AvgPool2d(kernel_size=r, stride=r)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Linear(dim, dim)
        self.DWconv = nn.Conv2d(in_channels=dim , out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.DWconvx = nn.Conv2d(in_channels=dim , out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.PWconv1 = nn.Conv2d(in_channels=dim , out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.windowatt = Attentions.WindowsSparseAttention(self.scale,r,num_heads)

        self.norm = nn.LayerNorm(dim, eps=1e-6)
    def forward(self, x):
        """
        args:
            x: NHWC tensor
        return:
            NHWC tensor
        """
        y = x
        x = torch.transpose(x, 1, 3)  # 将channels移动到第二个维度
        x = torch.transpose(x, 2, 3)  # 将height移动到第三个维度
        x = self.downsample(x)
        x = torch.transpose(x, 2, 3)  # 将height移动回第二个维度
        x = torch.transpose(x, 1, 3)  # 将channels移动回最后一个维度
        _, H, W, _ = x.size()
        x = rearrange(x, 'n h w c -> n (h w) c')

        #######################################
        B, N, C = x.shape  # 1,196,64

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        indices = torch.topk(attn, self.topk, dim=-1, largest=True).indices
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # print(f"x.shape = {x.shape}")
        x = rearrange(x, 'n (h w) c -> n h w c', h=H, w=W)

        _, H, W, _ = y.size()
        y = rearrange(y, 'n h w c -> n (h w) c')
        B, N, C = y.shape  # 1,196,64
        QKV = self.qkv(y).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        Q, K, V = QKV[0], QKV[1], QKV[2]
        Q = Q.permute(0, 1, 2, 3).reshape(B, self.num_heads, H, W, C // self.num_heads)
        K = K.permute(0, 1, 2, 3).reshape(B, self.num_heads, H, W, C // self.num_heads)
        V = V.permute(0, 1, 2, 3).reshape(B, self.num_heads, H, W, C // self.num_heads)
        y = self.windowatt(Q, K, V, indices)

        x = torch.transpose(x, 1, 3)  # 将channels移动到第二个维度
        x = torch.transpose(x, 2, 3)  # 将height移动到第三个维度
        x = F.interpolate(x, scale_factor=self.r, mode='nearest')
        x = self.DWconvx(x)
        x = torch.transpose(x, 2, 3)  # 将height移动回第二个维度
        x = torch.transpose(x, 1, 3)  # 将channels移动回最后一个维度
        #
        # y = self.DWconvy(y.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        z = torch.add(0.5*x, 0.5*y)
        z = z.permute(0, 3, 1, 2)

        z = self.DWconv(z)

        z = self.PWconv1(z)
        #z = self.PWconv1(z)
        z = z.permute(0, 2, 3, 1)
        return z
