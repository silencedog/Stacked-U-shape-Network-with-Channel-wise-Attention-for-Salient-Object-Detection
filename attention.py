import torch
import torch.nn as nn


class CAM_Module(nn.Module):

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = torch.sigmoid(energy_new)
        out = torch.bmm(attention, proj_query)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out * x

        return out
