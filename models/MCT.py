import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_blocks import *
import numpy as np
import cv2
import math
from models.Transformer import TransformerModel
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d




# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


##################################
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.h_transform(x)
        x3 = self.deconv3(x3)
        x3 = self.inv_h_transform(x3)
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x


    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

#################################################################
class MCT(nn.Module):
    def __init__(self,
                 arch,
                 scale_ratio,
                 n_select_bands,
                 n_bands,
                 dataset=None
                 ):
        """Load the pretrained ResNet and replace top fc layer."""
        super(MCT, self).__init__()

        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.arch = arch
        self.n_select_bands = n_select_bands
        self.weight = nn.Parameter(torch.tensor([0.5]))

        self.conv_fus = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_spat = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_spec = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_select_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.D1 = nn.Sequential(
            nn.Conv2d(n_select_bands, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, n_bands, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

        )
        self.D2 = nn.Sequential(
            nn.Conv2d(n_bands, 156, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, 156, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(156, n_bands*2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands*2, n_bands*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        #Pavia(n_bands*3-2) PaviaU(n_bands*3-1)

        if dataset == 'Pavia':
            u1_channel = n_bands*3-2
        elif dataset == 'PaviaU':
            u1_channel = n_bands * 3 - 1
        elif dataset == 'Botswana':
            u1_channel = n_bands * 3 - 3
        elif dataset == 'Urban':
            u1_channel = n_bands * 3 - 2
        elif dataset == 'Washington':
            u1_channel = n_bands * 3-1
        self.U1 = nn.Sequential(
            nn.Conv2d(u1_channel, n_bands*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands*2, n_bands*1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Pavia(n_bands*2) PaviaU(n_bands*2-2)
        if dataset == 'Pavia':
            u2_channel = n_bands * 2
        elif dataset == 'PaviaU':
            u2_channel = n_bands * 2 - 2
        elif dataset == 'Botswana':
            u2_channel = n_bands * 2 - 2
        elif dataset == 'Urban':
            u2_channel = n_bands * 2
        elif dataset == 'Washington':
            u2_channel = n_bands*2-2
        self.U2 = nn.Sequential(
            nn.Conv2d( u2_channel, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(n_bands*2+5, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.transformer1 = TransformerModel(
            map_size=8,
            M_channel = n_bands*2,
            dim=128,
            depth=5,
            heads=8,
            mlp_dim=n_bands,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
        )
        self.transformer2 = TransformerModel(
            map_size = 32,
            M_channel=n_bands,
            dim=64,
            depth=5,
            heads=8,
            mlp_dim=n_bands,
            dropout_rate=0.1,
            attn_dropout_rate=0.1
        )

        self.ca = ChannelAttention(2 * n_bands)
        self.ca1 = ChannelAttention( n_bands)
        self.sa = SpatialAttention()
        self.decoder1 = DecoderBlock(n_bands*6, n_bands*6, nn.BatchNorm2d)
        self.decoder2 = DecoderBlock(n_bands * 4, n_bands * 4, nn.BatchNorm2d)
        self.decoder3 = DecoderBlock(n_bands * 1, n_bands * 1, nn.BatchNorm2d)


    def lrhr_interpolate(self, x_lr, x_hr):
        x_lr = F.interpolate(x_lr, scale_factor=self.scale_ratio, mode='bilinear')
        gap_bands = self.n_bands / (self.n_select_bands - 1.0)
        for i in range(0, self.n_select_bands - 1):
            x_lr[:, int(gap_bands * i), ::] = x_hr[:, i, ::]
        x_lr[:, int(self.n_bands - 1), ::] = x_hr[:, self.n_select_bands - 1, ::]

        return x_lr

    def spatial_edge(self, x):
        edge1 = x[:, :, 0:x.size(2) - 1, :] - x[:, :, 1:x.size(2), :]
        edge2 = x[:, :, :, 0:x.size(3) - 1] - x[:, :, :, 1:x.size(3)]

        return edge1, edge2

    def spectral_edge(self, x):
        edge = x[:, 0:x.size(1) - 1, :, :] - x[:, 1:x.size(1), :, :]

        return edge


    def forward(self, x_lr, x_hr):


        if self.arch == 'MCT':

            a = self.D1(x_hr)
            a = a * self.ca1(a)
            a = a*self.sa(a)

            b = self.D2(a)
            b = b * self.ca(b)
            b = b * self.sa(b)

            c = self.D2(x_lr)
            c = c * self.ca(c)
            c = c * self.sa(c)
            d = F.interpolate(x_lr, scale_factor=4, mode='bilinear')
            d = d * self.ca1(d)
            d = d * self.sa(d)
#######################################################################################
            transformer_results = self.transformer1(b, c)
            e = transformer_results['z']
            f1 = torch.cat((torch.cat((b,c), 1), e),1)
            f1 = self.decoder1(f1)
            f1 = F.interpolate(f1, scale_factor=4, mode='bilinear')
            #f1_channel = f1.shape[1]
            f1 = self.U1(f1)
 ###################################################################################
            transformer_results1 = self.transformer2(a,x_lr)
            g = transformer_results1['z']
            f2 = torch.cat((torch.cat((a,x_lr),1),g),1)
            f2 = torch.cat((f2,f1),1)
            f2 = self.decoder2(f2)
            f2 = F.interpolate(f2, scale_factor=4, mode='bilinear')

            f2 = self.U2(f2)

            x = torch.cat((f2, x_hr), 1)
            x = torch.cat((x, d), 1)
            x = self.conv3(x)
            x_spat = x + self.conv_spat(x)
            spat_edge1, spat_edge2 = self.spatial_edge(x_spat)

            x_spec = x_spat + self.conv_spec(x_spat)
            spec_edge = self.spectral_edge(x_spec)

            x = x_spec
        return x, x_spat, x_spec, spat_edge1, spat_edge2, spec_edge
