import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.rep_dim = 32
        # GCN layers
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 48)
        self.conv3 = GCNConv(48, 64)
        # Output
        self.out = Linear(64*2, self.rep_dim)

    def forward(self, x, edge_index, batch_index):
        # Conv Layers
        hidden = F.relu(self.conv1(x, edge_index))
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.relu(hidden)
        # Global Pooling
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)
        out = self.out(hidden)
        return out

model = GCN(37)

