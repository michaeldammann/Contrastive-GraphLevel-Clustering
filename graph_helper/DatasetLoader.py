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

    def edgenumber_mapper(self, edgenumber):
        if edgenumber == 0:
            return 0
        elif edgenumber in range(1,20):
            return 1
        elif edgenumber in range(20,40):
            return 2
        elif edgenumber in range(40,60):
            return 3
        elif edgenumber in range(60,80):
            return 4
        elif edgenumber in range(80,100):
            return 5
        elif edgenumber in range(100,120):
            return 6
        elif edgenumber in range(120,140):
            return 7

    def __load_dataset(self, dataset):
        all_graphs_t = []
        all_y = []

        if dataset.lower()[0:11] == 'ceilographs': #'ceilographs4/8/16_year/month/location/season/nodenumber/none'
            graphfilename = '/home/rigel/MDammann/PycharmProjects/CC4Graphs/ceilo_graph_generation/graphs_processed.pickle'

            # load graphs
            with open(graphfilename, 'rb') as f:
                graphdict = pickle.load(f)

            for k, v in graphdict.items():
                all_graphs_t.extend(v)


            year_class_map = {"2014": 0, "2015": 1, "2016": 2, "2017": 3, "2018": 4}
            month_class_map = {"01": 0, "02": 1, "03": 2, "04": 3, "05": 4, "06": 5, "07": 6, "08": 7, "09": 8, "10": 9, "11":10, "12":11}
            seasons_class_map = {"01": 0, "02": 0, "03": 1, "04": 1, "05": 1, "06": 2, "07": 2, "08": 2, "09": 3, "10": 3,
                               "11": 3, "12": 0}
            location_class_map = {"hohenpeissenberg":0, "aachen":1, "hamburg":2, "hoyerswerda":3}
            nodenumber_class_map = {1:0,
                                    2:1, 3:1, 4:1, 5:1,
                                    6:2, 7:2, 8:2, 9:2,
                                    10:3, 11:3, 12:3, 13:3, 14:3,
                                    15:4, 16:4, 17:4, 18:4, 19:4,
                                    20:5, 21:5, 22:5, 23:5, 24:5,
                                    25:6, 26:6, 27:6, 28:6, 29:6,
                                    30:7, 31:7, 32:7}


            '''
            all_keys = graphdict.keys()
            all_years = [filename[0:4] for filename in all_keys]
            all_months = [filename[4:6] for filename in all_keys]
            split_keys = [filename.split("_") for filename in all_keys]
            all_locations = [split_key[1] for split_key in split_keys]

            all_years_classes = [year_class_map[year] for year in all_years]
            all_months_classes = [month_class_map[month] for month in all_months]
            all_locations_classes = [location_class_map[location] for location in all_locations]
            all_seasons_classes = [seasons_class_map[month] for month in all_months]
            '''

            dataset_mode = dataset.split('_')[1]

            for k, v in graphdict.items():
                k_year = k[0:4]
                k_month = k[4:6]
                split_key = k.split("_")
                k_location = split_key[1]
                for v_n in v:
                    if dataset_mode == 'year':
                        all_y.append(year_class_map[k_year])
                    elif dataset_mode == 'month':
                        all_y.append(month_class_map[k_month])
                    elif dataset_mode == 'location':
                        all_y.append(location_class_map[k_location])
                    elif dataset_mode == 'season':
                        all_y.append(seasons_class_map[k_month])
                    elif dataset_mode == 'nodenumber':
                        all_y.append(nodenumber_class_map[len(v_n.x.cpu().detach().numpy().tolist())])
                    elif dataset_mode == 'edgenumber':
                        all_y.append(self.edgenumber_mapper(int(len(v_n.edge_index.cpu().detach().numpy().tolist()[0])/2)))
                    elif dataset_mode == "features":
                        all_y.append(np.argmax(np.mean(v_n.x.cpu().detach().numpy()[:,:12], axis=0)))
                    else:
                        all_y.append(0)


        elif dataset in ['constructedgraphs_2', 'constructedgraphs_2nodedeg', 'constructedgraphs_4', 'constructedgraphs_2size', 'constructedgraphs_2features',
                         'constructedgraphs_4nodedeg', 'constructedgraphs_4size']:
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