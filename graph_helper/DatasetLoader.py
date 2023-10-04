import networkx
import torch
from networkx import diameter, average_clustering
import pickle
from torch_geometric.utils.convert import to_networkx
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
import torch_geometric
from torch_geometric.datasets import TUDataset, MNISTSuperpixels, ShapeNet, ModelNet, CoMA, GeometricShapes
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import ConcatDataset
import torch_geometric.transforms as T
from modules.graph_utils import max_degree_undirected


class DatasetLoader:

    def __init__(self, dataset_savedir):
        self.dataset_savedir = dataset_savedir

    def edgenumber_mapper(self, edgenumber):
        if edgenumber == 0:
            return 0
        elif edgenumber in range(1, 20):
            return 1
        elif edgenumber in range(20, 40):
            return 2
        elif edgenumber in range(40, 60):
            return 3
        elif edgenumber in range(60, 80):
            return 4
        elif edgenumber in range(80, 100):
            return 5
        elif edgenumber in range(100, 120):
            return 6
        elif edgenumber in range(120, 140):
            return 7

    def __load_dataset(self, dataset):
        all_graphs_t = []
        all_y = []

        if dataset == 'ogbg-molhiv':
            dataset = PygGraphPropPredDataset(root=self.dataset_savedir,
                                              name='ogbg-molhiv')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr,
                                                   y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0, 0])

        elif dataset == 'ogbg-ppa':
            dataset = PygGraphPropPredDataset(root=self.dataset_savedir,
                                              name='ogbg-ppa')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            max_deg = max_degree_undirected(dataset)
            transform = torch_geometric.transforms.OneHotDegree(max_degree=max_deg)
            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr,
                                                   y=data_b.y)
                all_graphs_t.append(transform(data_g))
                all_y.append(data_b.y.cpu().detach().numpy()[0, 0])


        elif dataset in ['TWITTER-Real-Graph-Partial', 'Yeast']:
            dataset = TUDataset(root=self.dataset_savedir,
                                name=dataset)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr,
                                                   y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])
        elif dataset in ['PROTEINS', 'DD']:
            dataset = TUDataset(root=self.dataset_savedir,
                                             name=dataset)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr,
                                                   y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])
        elif dataset in ['reddit_threads', 'twitch_egos', 'REDDIT-BINARY', 'TRIANGLES']:
            dataset_pretransform = TUDataset(root=self.dataset_savedir,
                                             name=dataset)
            dataset = TUDataset(root=self.dataset_savedir, name=dataset,
                                transform=torch_geometric.transforms.OneHotDegree(
                                    max_degree=max_degree_undirected(dataset_pretransform)))
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr,
                                                   y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])
        elif dataset == 'MNISTSuperpixels':
            dataset_train = MNISTSuperpixels(root=self.dataset_savedir,
                                             train=True)
            loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)
            dataset_test = MNISTSuperpixels(root=self.dataset_savedir,
                                            train=False)
            loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

            it = iter(loader_train)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr,
                                                   y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])

            it = iter(loader_test)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr,
                                                   y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])

        return all_graphs_t, all_y

    def load_dataset_ptg(self, dataset):
        return self.__load_dataset(dataset)

    def load_dataset_nx(self, dataset):
        graphs, all_y = self.__load_dataset(dataset)
        return [to_networkx(graph, to_undirected=True, node_attrs=["x"]) for graph in graphs], all_y

    def load_dataset_nx_g2v(self, dataset):
        graphs, all_y = self.__load_dataset(dataset)
        ds_nx = [to_networkx(graph, to_undirected=True, node_attrs=["x"]) for graph in graphs]
        for g in ds_nx:
            for tpl in g.nodes(data=True):
                tpl[1]["feature"] = tpl[1].pop("x")
        return ds_nx, all_y
