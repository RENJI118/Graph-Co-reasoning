import torch
import torch.nn as nn
import torch.nn.functional as F



norm_dict = {'BATCH': nn.BatchNorm2d, 'INSTANCE': nn.InstanceNorm2d, 'GROUP': nn.GroupNorm}


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky=True, norm='BATCH', activation=True, deform=False):
        super().__init__()
        # determine basic attributes
        self.norm_type = norm
        self.activation = activation
        self.leaky = leaky
        padding = (kernel_size - 1) // 2

        # activation, support PReLU and common ReLU
        if self.leaky:
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

        # instantiate layers

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        norm_layer = norm_dict[norm]
        if norm in ['BATCH', 'INSTANCE']:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = norm_layer(8, in_channels)

    def basic_forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.act(x)
        return x

    def group_forward(self, x):
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

    def forward(self, x):
        if self.norm_type in ['BATCH', 'INSTANCE']:
            return self.basic_forward(x)
        else:
            return self.group_forward(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, leaky=False, lkdw=False, norm='BATCH', deform=False):
        super().__init__()
        self.norm_type = norm

        self.act = nn.ReLU(inplace=True)
        self.conv1 = ConvNorm(in_channels, out_channels, 3, stride, leaky, norm, True, deform)
        self.conv2 = ConvNorm(out_channels, out_channels, 3, 1, leaky, norm, False)

        self.identity_mapping = ConvNorm(in_channels, out_channels, 1, stride, leaky, norm, False)

        self.need_map = in_channels != out_channels or stride != 1

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.need_map:
            identity = self.identity_mapping(identity)

        out = out + identity
        if self.norm_type != 'GROUP':
            out = self.act(out)

        return out

