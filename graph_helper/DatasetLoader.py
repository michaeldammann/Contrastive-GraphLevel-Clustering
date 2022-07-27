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

        elif dataset in ['constructedgraphs_2', 'constructedgraphs_2nodedeg', 'constructedgraphs_4', 'constructedgraphs_2size', 'constructedgraphs_2features']:
            with open('/home/rigel/MDammann/PycharmProjects/CC4Graphs/baselines/{}.pkl'.format(dataset), 'rb') as handle:
                constructedgraphs = pickle.load(handle)

            for graph in constructedgraphs:
                data_g = torch_geometric.data.Data(x=torch.tensor(graph[0]), edge_index=torch.tensor(graph[1]), y=torch.tensor(graph[2]))
                all_graphs_t.append(data_g)
                all_y.append(graph[2])

        elif dataset == 'ogbg-molhiv':
            dataset = PygGraphPropPredDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='ogbg-molhiv')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0,0])

        elif dataset == 'ogbg-ppa':
            dataset = PygGraphPropPredDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='ogbg-ppa')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            max_deg = max_degree_undirected(dataset)
            transform = torch_geometric.transforms.OneHotDegree(max_degree=max_deg)
            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(transform(data_g))
                all_y.append(data_b.y.cpu().detach().numpy()[0,0])

        elif dataset == 'ogbg-ppa10000':
            dataset = PygGraphPropPredDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='ogbg-ppa')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                if len(all_graphs_t)>10000:
                    break
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0,0])

        elif dataset == 'TwitchVsDeezerEgos_balanced':
            dataset_deezer_pretransform = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/',
                                             name='deezer_ego_nets')
            dataset_twitch_pretransform = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/',
                                             name='twitch_egos')
            max_degree = max(max_degree_undirected(dataset_deezer_pretransform), max_degree_undirected(dataset_twitch_pretransform))

            dataset_deezer = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='deezer_ego_nets',
                                transform=torch_geometric.transforms.OneHotDegree(
                                    max_degree=max_degree))
            loader_deezer = DataLoader(dataset_deezer, batch_size=1, shuffle=False)

            it = iter(loader_deezer)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(0)


            dataset_twitch = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='twitch_egos',
                                transform=torch_geometric.transforms.OneHotDegree(
                                    max_degree=max_degree))
            loader_twitch = DataLoader(dataset_twitch, batch_size=1, shuffle=False)

            it = iter(loader_twitch)
            counter = 0
            max_len = len(all_graphs_t)
            for data_b in it:
                if counter > max_len:
                    break
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(1)
                counter+=1

        elif dataset == 'TwitchVsGithub_balanced':
            dataset_github_pretransform = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/',
                                             name='github_stargazers')
            dataset_twitch_pretransform = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/',
                                             name='twitch_egos')
            max_degree = max(max_degree_undirected(dataset_github_pretransform), max_degree_undirected(dataset_twitch_pretransform))

            dataset_github = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='github_stargazers',
                                transform=torch_geometric.transforms.OneHotDegree(
                                    max_degree=max_degree))
            loader_github = DataLoader(dataset_github, batch_size=1, shuffle=False)

            it = iter(loader_github)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(0)


            dataset_twitch = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='twitch_egos',
                                transform=torch_geometric.transforms.OneHotDegree(
                                    max_degree=max_degree))
            loader_twitch = DataLoader(dataset_twitch, batch_size=1, shuffle=False)

            it = iter(loader_twitch)
            counter = 0
            max_len = len(all_graphs_t)
            for data_b in it:
                if counter > max_len:
                    break
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(1)
                counter+=1


        elif dataset in ['TWITTER-Real-Graph-Partial', 'Yeast']:
            dataset = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/',
                                name=dataset)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])
        elif dataset in ['reddit_threads', 'twitch_egos', 'REDDIT-BINARY', 'TRIANGLES']:
            dataset_pretransform = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/',
                                             name=dataset)
            dataset = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name=dataset,
                                transform=torch_geometric.transforms.OneHotDegree(
                                    max_degree=max_degree_undirected(dataset_pretransform)))
            loader = DataLoader(dataset, batch_size=1, shuffle=False)

            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])
        elif dataset == 'ShapeNet':
            dataset = ShapeNet(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/ShapeNet')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])
        elif dataset in ['ModelNet10', 'ModelNet40']:
            dataset = ModelNet(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/'+dataset, name=dataset[-2:], pre_transform=T.FaceToEdge)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])
        elif dataset == 'CoMA':
            dataset = CoMA(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/'+dataset, pre_transform=T.FaceToEdge)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            it = iter(loader)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
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
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])

            it = iter(loader_test)
            for data_b in it:
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
                all_graphs_t.append(data_g)
                all_y.append(data_b.y.cpu().detach().numpy()[0])

        elif dataset == 'MNISTSuperpixels500':
            dataset_train = MNISTSuperpixels(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', train=True)
            loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

            it = iter(loader_train)
            for data_b in it:
                if len(all_graphs_t)>=500:
                    break
                data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr, y=data_b.y)
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