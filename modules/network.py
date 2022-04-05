import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, gnn, feature_dim, class_num):
        super(Network, self).__init__()
        self.gnn = gnn
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.Linear(self.gnn.rep_dim, self.gnn.rep_dim),
            nn.ReLU(),
            nn.Linear(self.gnn.rep_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.gnn.rep_dim, self.gnn.rep_dim),
            nn.ReLU(),
            nn.Linear(self.gnn.rep_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        '''
        print('x_i')
        print(x_i)
        print(x_i.x.float())
        print(x_i.x.float().cpu().detach().numpy().shape)
        print(x_i.edge_index)
        print(x_i.edge_index.cpu().detach().numpy().shape)
        print(x_i.batch)
        print(x_i.batch.cpu().detach().numpy().shape)
        print('x_j')
        print(x_j)
        print(x_j.x.float())
        print(x_j.x.float().cpu().detach().numpy().shape)
        print(x_j.edge_index)
        print(x_j.edge_index.cpu().detach().numpy().shape)
        print(x_j.batch)
        print(x_j.batch.cpu().detach().numpy().shape)
        '''
        h_i = self.gnn(x_i.x.float(), x_i.edge_index, x_i.batch)
        h_j = self.gnn(x_j.x.float(), x_j.edge_index, x_j.batch)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.resnet(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
