import torch
import torch.nn as nn
import math
from model.layer import ConvBnRelu, UBlock, conv1x1

class channel_attention(nn.module):
    def __init__(self, channel, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class eca_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

class CrossModalityInterConvModule(nn.Module):

    def __init__(self, num_channels, norm_layer, dropout, dilation=(1, 1), reduction_ratio=2):
        super(CrossModalityInterConvModule, self).__init__()
        self.modality_branch_channels = num_channels // 2
        self.branch_att_weight_func = CrossModalityBranchAttention(num_channels=num_channels,
                                                                reduction_ratio=reduction_ratio)

        self.modality_branch_1_conv = ConvBnRelu(self.modality_branch_channels, self.modality_branch_channels,
                                                 norm_layer, dilation=dilation[0], dropout=dropout)
        self.modality_branch_2_conv = ConvBnRelu(self.modality_branch_channels, self.modality_branch_channels,
                                                 norm_layer, dilation=dilation[0], dropout=dropout)

    def forward(self,input_tensor):
        # pairing
        x=input_tensor
        input_modality_branch_1_tensor, input_modality_branch_2_tensor = torch.chunk(input_tensor, 2, dim=1)

        branch_1_channel_weight, branch_2_channel_weight = self.branch_att_weight_func(input_tensor)
        conv_modality_branch_1_tensor = self.modality_branch_1_conv(input_modality_branch_1_tensor)
        conv_modality_branch_2_tensor = self.modality_branch_2_conv(input_modality_branch_2_tensor)


class CrossModalityBranchAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):

        super(CrossModalityBranchAttention, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio

        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2_branch_1 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.fc2_branch_2 = nn.Linear(num_channels_reduced, num_channels, bias=True)

    def forward(self,input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        avg_squeeze_tensor = self.global_avg_pool(input_tensor)
        max_squeeze_tensor = self.global_max_pool(input_tensor)

        #avg
        avg_fc_out_1 = self.fc1(avg_squeeze_tensor.view(batch_size, num_channels))
        max_fc_out_1 = self.fc1(max_squeeze_tensor.view(batch_size, num_channels))\

        shared_fc_out_1 = avg_fc_out_1 + max_fc_out_1

        branch_1_attention_weight = self.fc2_branch_1(shared_fc_out_1)
        branch_2_attention_weight = self.fc2_branch_2(shared_fc_out_1)

        branch_1_attention_weight = self.softmax(branch_1_attention_weight)
        branch_2_attention_weight = self.softmax(branch_2_attention_weight)

        branch_1_ca_weight = branch_1_attention_weight.view(batch_size, num_channels, 1, 1)
        branch_2_ca_weight = branch_2_attention_weight.view(batch_size, num_channels, 1, 1)

        return branch_1_ca_weight, branch_2_ca_weight