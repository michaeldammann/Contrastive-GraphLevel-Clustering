import networkx
import torch
from networkx import diameter, average_clustering
import pickle
from torch_geometric.utils.convert import to_networkx
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
import torch_geometric
from torch_geometric.datasets import TUDataset, MNISTSuperpixels
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import ConcatDataset

class DatasetLoader:

    def __load_dataset(self, dataset):
        all_graphs_t = []
        all_y = []

        if dataset == 'CeiloGraphs':
            graphfilename = '/home/rigel/MDammann/PycharmProjects/CC4Graphs/ceilo_graph_generation/graphs_processed.pickle'

            # load graphs
            with open(graphfilename, 'rb') as f:
                graphdict = pickle.load(f)

            for k, v in graphdict.items():
                all_graphs_t.extend(v)

        elif dataset == 'ConstructedGraphs':
            with open('/home/rigel/MDammann/PycharmProjects/CC4Graphs/baselines/constructedgraphs.pkl', 'rb') as handle:
                constructedgraphs = pickle.load(handle)

            for graph in constructedgraphs:
                data_g = torch_geometric.data.Data(x=torch.tensor(graph[0]), edge_index=torch.tensor(graph[1]))
                all_graphs_t.append(data_g)
                all_y.append(graph[2])

        elif dataset == 'ogbg-molhiv':
            dataset = PygGraphPropPredDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='ogbg-molhiv')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0,0])

        elif dataset == 'ogbg-ppa':
            dataset = PygGraphPropPredDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='ogbg-ppa')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0,0])

        elif dataset == 'ogbg-ppa10000':
            dataset = PygGraphPropPredDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='ogbg-ppa')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                if len(all_graphs_t)>10000:
                    break
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr)
                print(data_g)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0,0])

        elif dataset == 'TWITTER-Real-Graph-Partial':
            dataset = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/',
                                name='TWITTER-Real-Graph-Partial')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])

        elif dataset == 'MNISTSuperpixels':
            dataset_train = MNISTSuperpixels(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', train=True)
            loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)
            dataset_test = MNISTSuperpixels(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/',
                                             train=False)
            loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

            it = iter(loader_train)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])

            it = iter(loader_test)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])

        elif dataset == 'MNISTSuperpixels500':
            dataset_train = MNISTSuperpixels(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', train=True)
            loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

            it = iter(loader_train)
            for data_b in it:
                if len(all_graphs_t)>=500:
                    break
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])

        return all_graphs_t, all_y



    def load_dataset_ptg(self, dataset):
        return self.__load_dataset(dataset)

    def load_dataset_nx(self, dataset):
        graphs, all_y = self.__load_dataset(dataset)
        return [to_networkx(graph, to_undirected=True, node_attrs=["x"]) for graph in graphs], all_y