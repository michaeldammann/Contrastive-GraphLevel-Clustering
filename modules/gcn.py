from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import ChebConv, GCNConv  # noqa


dataset_name = "NCI1"

dataset = TUDataset(root='/home/md/PycharmProjects/CC4Graphs/datasets/', name=dataset_name)
loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


