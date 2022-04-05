import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool as gap, global_max_pool as gmp
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.rep_dim = 128
        # GCN layers
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 64)
        self.conv5 = GCNConv(64, 128)
        # Output
        self.lin_before_pool = Linear(128, 128)
        self.lin_after_pool = Linear(128 * 2, 128 * 2)
        self.out = Linear(128*2, self.rep_dim)

    def forward(self, x, edge_index, batch_index):
        # Conv Layers
        hidden = F.relu(self.conv1(x, edge_index))
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv4(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.conv5(hidden, edge_index)
        hidden = F.relu(hidden)
        hidden = self.lin_before_pool(hidden)
        # Global Pooling
        hidden = torch.cat([gmp(hidden, batch_index),
                            gap(hidden, batch_index)], dim=1)
        hidden = self.lin_after_pool(hidden)
        out = self.out(hidden)
        return out
