import torch.nn as nn
from models.IntmdSequential import IntermediateSequential
import numpy as np
import torch

#######################################################################################################
class SelfAttentionsingle(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.out = {}

    def forward(self, input):
        # print(x.shape) #1, 4, 128, 128, 128
        # x_ = input['x'].permute(0, 2, 3, 1).contiguous()
        # y_ = input['y'].permute(0, 2, 3, 1).contiguous()
        # # x = x.permute(0, 2, 3, 1).contiguous()
        # x = x_.view(x_.size(0), x_.size(2)*x_.size(1), -1)
        # y = y_.view(y_.size(0), y_.size(2) * y_.size(1), -1)
        # a = x = self.linear_encoding(x)
        # b = y = self.linear_encoding(y)

        a = x = input['x']
       # b = y = input['y']

        # x = np.squeeze(x, axis=0)
        # y = np.squeeze(y, axis=0)

        B, N, C = x.shape
        qkv1 = (
            self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        # qkv2 = (
        #     self.qkv(y)
        #         .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #         .permute(2, 0, 3, 1, 4)
        # )
        q, k, v = (
            qkv1[0],
            qkv1[1],
            qkv1[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        # q1, k1, v1 = (
        #     qkv2[0],
        #     qkv2[1],
        #     qkv2[2],
        # )  # ma
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        # attn1 = attn1.softmax(dim=-1)
        # attn1 = self.attn_drop(attn1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + a
        # y = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
        #         # y = self.proj(y)
        #         # y = self.proj_drop(y)
        #         # y = y + b

        self.out['x']= x
        return self.out
############################################################################################################
class SelfAttention(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.out = {}

    def forward(self, input):
        # print(x.shape) #1, 4, 128, 128, 128
        # x_ = input['x'].permute(0, 2, 3, 1).contiguous()
        # y_ = input['y'].permute(0, 2, 3, 1).contiguous()
        # # x = x.permute(0, 2, 3, 1).contiguous()
        # x = x_.view(x_.size(0), x_.size(2)*x_.size(1), -1)
        # y = y_.view(y_.size(0), y_.size(2) * y_.size(1), -1)
        # a = x = self.linear_encoding(x)
        # b = y = self.linear_encoding(y)

        a = x = input['x']
        b = y = input['y']

        # x = np.squeeze(x, axis=0)
        # y = np.squeeze(y, axis=0)

        B, N, C = x.shape
        qkv1 = (
            self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        qkv2 = (
            self.qkv(y)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv1[0],
            qkv1[1],
            qkv1[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        q1, k1, v1 = (
            qkv2[0],
            qkv2[1],
            qkv2[2],
        )  # ma
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        x = (attn @ v1).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + a
        y = (attn1 @ v).transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        y = y + b

        self.out['x'], self.out['y'] = x, y
        return self.out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.input = {}

    def forward(self, x):
        self.input['x'] = self.norm(x['x'])
        self.input['y'] = self.norm(x['y'])
        return self.fn(self.input)


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn
        self.input = {}
        self.out = {}

    def forward(self, x):
        self.input['x'] = self.norm(x['x'])
        self.input['y'] = self.norm(x['y'])
        self.out['x'], self.out['y'] = self.dropout(self.fn(self.input)['x']), self.dropout(self.fn(self.input)['y'])
        return self.out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),改了
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )
        self.out = {}

    def forward(self, input):
        self.out['x'], self.out['y'] = self.net(input['x']), self.net(input['y'])
        return self.out


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) #8x

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings

#############################################################################
class TransformerModelsingle(nn.Module):
    def __init__(
            self,
            map_size,
            M_channel,
            dim,
            depth,
            heads,
            mlp_dim,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)
        self.input = {}
        self.output = {}
        self.map_size = map_size
        self.linear_encoding = nn.Linear(M_channel, dim)
        self.linear_encoding_de = nn.Linear(dim, M_channel)
        self.position_encoding = LearnedPositionalEncoding(M_channel, dim, map_size*map_size)


    def forward(self, x):
        # self.input['x'] = x
        # self.input['y'] = y

        x_ = x.permute(0, 2, 3, 1).contiguous()

        x = x_.view(x_.size(0), x_.size(2)*x_.size(1), -1)

        self.input['x'] = self.position_encoding(self.linear_encoding(x))

        results = self.net(self.input)
        x = results['x']
        x = self.linear_encoding_de(x).permute(0, 2, 1).contiguous()
        self.output['x'] = x.view(x.size(0), x.size(1), self.map_size, self.map_size)
        return self.output['x']
############################################################################################

class TransformerModel(nn.Module):
    def __init__(
            self,
            map_size,
            M_channel,
            dim,
            depth,
            heads,
            mlp_dim,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)
        self.input = {}
        self.output = {}
        self.map_size = map_size
        self.linear_encoding = nn.Linear(M_channel, dim)
        self.linear_encoding_de = nn.Linear(dim, M_channel)
        self.position_encoding = LearnedPositionalEncoding(M_channel, dim, map_size*map_size)
        #self.transformersingle = TransformerModelsingle()

    def forward(self, x, y):
        # self.input['x'] = x
        # self.input['y'] = y

        x_ = x.permute(0, 2, 3, 1).contiguous()
        y_ = y.permute(0, 2, 3, 1).contiguous()
        x = x_.view(x_.size(0), x_.size(2)*x_.size(1), -1)
        y = y_.view(y_.size(0), y_.size(2) * y_.size(1), -1)
        self.input['x'] = self.position_encoding(self.linear_encoding(x))
        self.input['y'] = self.position_encoding(self.linear_encoding(y))
        results = self.net(self.input)
        x, y = results['x'], results['y']
        x = self.linear_encoding_de(x).permute(0, 2, 1).contiguous()
        self.output['x'] = x.view(x.size(0), x.size(1), self.map_size, self.map_size)
        y = self.linear_encoding_de(y).permute(0, 2, 1).contiguous()
        self.output['y'] = y.view(y.size(0), y.size(1), self.map_size, self.map_size)
        self.output['z'] = self.output['x'] + self.output['y']
        #self.output['z'] = self.transformersingle(self.output['z'])
        return self.output
