import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from timm.models.vision_transformer import Block as TimmBlock


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class IWSA(nn.Module):
    """
    Interactive Window-based Self-Attention (IWSA), which utilize a Local Interactive Module (LIM) to merge
    information bewteen discrete windows.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        ws (int): The height and width of the window. Default: 7
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=7):
        assert ws != 1
        super(IWSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws
        self.lim_func = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def lim(self, x, func):
        """
        the local interactive module (LIM)
        Args:
            x (tensor): shape-intergrated feat maps, shape=(B, C, H, W)
            func: the function of LIM
        Returns:
            y_lim (tensor): interacted feat maps, shape=(B, H*W, C)
        """
        y_lim = func(x).flatten(2).transpose(1, 2)  # (B, H*W, C)
        return y_lim.contiguous()

    def img2win(self, x, H, W):
        """
        To split one img with shape of H*W to (H*W) / (H_sp*W_sp) windows.
        Args:
            x (tensor): shape=(B, H*W, C)
        Return:
            x (tensor): a group of windows, shape=(B, h_g*w_g, num_heads, ws*ws, head_dim)
        """
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
        # shape: (B, h_g*w_g, ws*ws, num_heads, head_dim) -> (B, h_g*w_g, num_heads, ws*ws, head_dim)
        x = x.reshape(B, h_group * w_group, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4)
        return x

    def win2img(self, x, H, W):
        """
        To merge a group of discrete windows to one shape-integrated feat map
        --x: a group of windows, shape=(B, h_g*w_g, num_heads, ws*ws, head_dim)
        return: a shape-integrated feat map, shape=(B, C, H', W')
        """
        h_group, w_group = H // self.ws, W // self.ws
        B = x.shape[0]
        C = x.shape[-1] * self.num_heads
        x = x.permute(0, 1, 3, 2, 4).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = x.transpose(2, 3).reshape(B, H, W, C)
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x, H, W):
        B, N, C = x.shape

        h_group, w_group = H // self.ws, W // self.ws

        #  TODO: revise to a clean one
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.img2win(q, H, W)
        k = self.img2win(k, H, W)
        v = self.img2win(v, H, W)

        v_shape_inter = self.win2img(v, H, W) # shape=(B, C, H, W)
        v_info_inter = self.lim(v_shape_inter, self.lim_func)  # shape=(B, N, C)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # shape=(B, h_g*w_g, n_head, ws*ws, ws*ws)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 
        # shape: (B, h_g*w_g, num_heads, ws*ws, head_dim) -> (B, h_g*w_g, ws*ws, num_heads, head_dim)
        # -> (B, h_g, w_g, ws, ws, num_heads*head_dim)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, N, C)
        x = x + v_info_inter
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SSA(nn.Module):
    """
    Scalable Self-Attention, which scale spatial and channel dimension to obtain a better trade-off.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        sr_ratio (float): spatial reduction ratio, varied with stages.
        c_ratio (float): channel ratio, varied with stages.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                sr_ratio=1.0, c_ratio=1.25):
        super(SSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.sr_ratio = sr_ratio
        self.c_new = int(dim * c_ratio) # scaled channel dimension
        print(f'@ dim: {dim}, dim_new: {self.c_new}, c_ratio: {c_ratio}, sr_ratio: {sr_ratio}\n')

        # self.scale = qk_scale or head_dim ** -0.5
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        if sr_ratio > 1:
            self.q = nn.Linear(dim, self.c_new, bias=qkv_bias)
            self.reduction = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim),
                nn.Conv2d(dim, self.c_new, kernel_size=1, stride=1))
            self.norm_act = nn.Sequential(
                nn.LayerNorm(self.c_new),
                nn.GELU())
            self.k = nn.Linear(self.c_new, self.c_new, bias=qkv_bias)
            self.v = nn.Linear(self.c_new, dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.sr_ratio > 1:
            # reduction
            _x = x.permute(0, 2, 1).reshape(B, C, H, W)
            _x = self.reduction(_x).reshape(B, self.c_new, -1).permute(0, 2, 1)  # shape=(B, N', C')
            _x = self.norm_act(_x)
            # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(N, N', C)
            q = self.q(x).reshape(B, N, self.num_heads, int(self.c_new / self.num_heads)).permute(0, 2, 1, 3)
            k = self.k(_x).reshape(B, -1, self.num_heads, int(self.c_new / self.num_heads)).permute(0, 2, 1, 3)
            v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
        else:
            q = self.q(x).reshape(B, N, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, int(C / self.num_heads)).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ScalableViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, c_ratio=1.25, sr_ratio=1, ws=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ws == 1:
            self.attn = SSA(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                c_ratio=c_ratio)
        else:
            self.attn = IWSA(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                ws=ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        # TODO: add scale like PoolFormer
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvPatchEmbed(nn.Module):
    """
    Convolutional Patch Embedding
    Args:
        img_size (int): input image / feat map size
        patch_size (int): pathch size
        in_chans (int): channels of input image / feat map
        embed_dim (int): the channel of tokens after token embedding
        overlapping (int): patch overlapping
        norm_layer (nn.Module): normalization layer
    Return:
        x: embeded feature map
        (H, W): the height and width of embeded feature map
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=64, overlapping=3, norm_layer=nn.LayerNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        kernel_size = to_2tuple(patch_size[0]+overlapping)
        padding = to_2tuple(overlapping)
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        print(f"ConvPatchEmbed: @input_channels: {in_chans}, output_channel: {embed_dim},\
             @kernel_size: {kernel_size}, @stride: {patch_size}, @padding: {padding}")
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size, patch_size, padding)
        self.norm = norm_layer(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class PosCNN(nn.Module):
    """
    This is PEG module from https://arxiv.org/abs/2102.10882
    """
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).contiguous().view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class ScalableViT(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 256],
            num_heads=[1, 2, 4],
            mlp_ratios=[4, 4, 4],
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[4, 4, 4],
            sr_ratios=[4, 2, 1],
            block_cls=ScalableViTBlock,
            wss=[7, 7, 7],
            c_ratios=[1.25, 1.25, 1.25, 1.25],
            plain_head=False):
        super().__init__()

        self.wss = wss
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.depths = depths

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        # transformer stages
        self.patch_embeds = nn.ModuleList()
        self.pos_block = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        self.stages = nn.ModuleList()
        
        for k in range(len(depths)):
            # patch embeddings
            if k == 0:
                self.patch_embeds.append(
                    ConvPatchEmbed(img_size, patch_size, in_chans,
                    embed_dims[k], overlapping=3, norm_layer=norm_layer))
            else:
                self.patch_embeds.append(
                    ConvPatchEmbed(img_size//patch_size//2**(k-1), 2, embed_dims[k-1],
                    embed_dims[k],  overlapping=1, norm_layer=norm_layer))
            # pos embeddings
            self.pos_block.append(PosCNN(embed_dims[k], embed_dims[k]))
            # pos drop
            self.pos_drops.append(nn.Dropout(p=drop_rate))
            # transformer blocks
            blocks = nn.ModuleList([
                block_cls(
                    dim=embed_dims[k],
                    num_heads=num_heads[k],
                    mlp_ratio=mlp_ratios[k],
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[k],
                    ws=1 if i % 2 == 1 else wss[k],
                    c_ratio=c_ratios[k]
                    ) for i in range(depths[k])])
            self.stages.append(blocks)
            cur += depths[k]
        # norm layer before head
        self.norm = norm_layer(embed_dims[-1])
        # head
        last_conv_out_channel = 4096
        self.head = nn.Sequential(
            nn.Linear(embed_dims[-1], last_conv_out_channel, bias=True),
            norm_layer(last_conv_out_channel),
            nn.GELU(),
            nn.Linear(last_conv_out_channel, num_classes) if num_classes > 0 else nn.Identity(),
        )
        if plain_head:
            self.reset_classifier(num_classes)
        
        self.apply(self._init_weights)
    
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
    
    def get_classifier(self):
        return self.head

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for k in range(len(self.depths)):
            for i in range(self.depths[k]):
                self.stages[k][i].drop_path.drop_prob = dpr[cur + i]
                if torch.cuda.current_device() == 0:
                    print('set drop_path', dpr[cur + i])
            cur += self.depths[k]
    
    @torch.jit.ignore
    def no_weight_decay(self):
        """this function will be used in create_optimizer of timm"""
        return set(['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.stages[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x.mean(dim=1)  # GAP her

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def scalable_vit_small(pretrained=False, **kwargs):
    img_size = 224
    model = ScalableViT(
        img_size=img_size,
        patch_size=4,
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 20, 2],
        wss=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        block_cls=ScalableViTBlock,
        c_ratios=[1.25, 1.25, 1.25, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def scalable_vit_small_0(pretrained=False, **kwargs):
    img_size = 224
    model = ScalableViT(
        img_size=img_size,
        patch_size=4,
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 20, 2],
        wss=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        block_cls=ScalableViTBlock,
        c_ratios=[1.25, 1.25, 1.25, 1.25],
        plain_head=True,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def scalable_vit_base(pretrained=False, **kwargs):
    img_size = 224
    model = ScalableViT(
        img_size=img_size,
        patch_size=4,
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 14, 6],
        wss=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        block_cls=ScalableViTBlock,
        c_ratios=[2, 1.25, 1.25, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def scalable_vit_large(pretrained=False, **kwargs):
    img_size = 224
    model = ScalableViT(
        img_size=img_size,
        patch_size=4,
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 6, 12, 4],
        wss=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        block_cls=ScalableViTBlock,
        c_ratios=[0.25, 0.5, 1, 1],
        **kwargs)
    model.default_cfg = _cfg()
    return model
