import torch
import torch.nn as nn
import torch.nn.functional as F

class SIGR(nn.Module):
    """
    Spatial Interaction Graph Reasoning
    """

    def __init__(self, planes):
        super(SIGR, self).__init__()

        # self.spatial_num_node = 8x(DWH)/2
        self.spatial_num_state = planes // 2

        self.downsampling = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=16, num_channels=planes),
        )

        self.node_k = nn.Conv2d(planes, self.spatial_num_state, kernel_size=1)
        self.node_v = nn.Conv2d(planes, self.spatial_num_state, kernel_size=1)
        self.node_q = nn.Conv2d(planes, self.spatial_num_state, kernel_size=1)

        self.conv_wg = nn.Conv1d(self.spatial_num_state, self.spatial_num_state, kernel_size=1, bias=False)
        self.bn_wg = nn.GroupNorm(num_groups=16, num_channels=self.spatial_num_state)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(self.spatial_num_state, planes, kernel_size=1),
                                  nn.GroupNorm(num_groups=16, num_channels=planes))

    def forward(self, input_feat):

        # Projection
        x = self.downsampling(input_feat)

        # V_s
        node_v = self.node_v(x)
        # Q_s
        node_q = self.node_q(x)
        # K_s
        node_k = self.node_k(x)

        b, c, h, w = node_v.size()

        # reshape
        node_v = node_v.view(b, c, -1)
        node_q = node_q.view(b, c, -1)
        node_k = node_k.view(b, c, -1)

        # transpose
        node_v = node_v.permute(0, 2, 1)
        node_q = node_q
        node_k = node_k.permute(0, 2, 1)

        # Reasoning
        # graph convolution
        A = self.softmax(torch.bmm(node_k, node_q))
        AV = torch.bmm(A, node_v)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = F.relu_(self.bn_wg(AVW))

        # Reprojection
        AVW = AVW.view(b, c, h, -1)
        sigr_out = self.out(AVW) + x

        F_sg = F.interpolate(sigr_out, size=input_feat.size()[2:], mode='bilinear', align_corners=False)

        # spatial gr output
        spatial_gr_out = F_sg + input_feat

        return spatial_gr_out
