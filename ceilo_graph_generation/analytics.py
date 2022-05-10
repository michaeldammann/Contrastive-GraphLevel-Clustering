import networkx
from networkx import diameter, average_clustering
import pickle
from torch_geometric.utils.convert import to_networkx
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
import torch_geometric
from torch_geometric.datasets import TUDataset
import numpy as np
from scipy.stats import sem

dataset = 'TWITTER-Real-Graph-Partial' #'ogbg-molhiv', 'CeiloGraphs', 'TWITTER-Real-Graph-Partial'

if dataset == 'CeiloGraphs':
    graphfilename = '/home/rigel/MDammann/PycharmProjects/CC4Graphs/ceilo_graph_generation/graphs_processed.pickle'

    #load graphs
    with open(graphfilename, 'rb') as f:
        graphdict = pickle.load(f)

    all_graphs_t = []

    for k, v in graphdict.items():
        all_graphs_t.extend(v)


elif dataset == 'ogbg-molhiv':
    dataset = PygGraphPropPredDataset(name = 'ogbg-molhiv')
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    all_graphs_t = []
    it = iter(loader)
    for data_b in it:
        data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr)
        all_graphs_t.append(data_g)

elif dataset == 'TWITTER-Real-Graph-Partial':
    dataset = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name='TWITTER-Real-Graph-Partial')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    all_graphs_t = []
    it = iter(loader)
    for data_b in it:
        data_g = torch_geometric.data.Data(x=data_b.x, edge_index=data_b.edge_index, edge_attr=data_b.edge_attr)
        all_graphs_t.append(data_g)


all_graphs_nx=[]
for graph in all_graphs_t:
    all_graphs_nx.append(to_networkx(graph, to_undirected=True))

# #graphs
print('#Graphs ',len(all_graphs_t))

print('#Node Features ', len(all_graphs_t[0].x[1])) #len(graph.x), 16 for CeiloGraphs

if dataset not in ['CeiloGraphs', 'DBLP_v1']:
    # #edge features
    print('#Edge Features ', len(all_graphs_t[0].edge_attr[1])) #len(graph.x), 16 for CeiloGraphs

# average #nodes per graph
all_n_nodes=[]
for graph in all_graphs_nx:
    all_n_nodes.append(graph.number_of_nodes())
print('Average #nodes: ', np.mean(all_n_nodes), ', std: ', np.std(all_n_nodes), ', sem: ', sem(all_n_nodes))


# average #edges per graph
all_n_edges=[]
for graph in all_graphs_nx:
    all_n_edges.append(graph.number_of_edges())
print('Average #edges: ', np.mean(all_n_edges), ', std: ', np.std(all_n_edges), ', sem: ', sem(all_n_edges))

# average node degree
all_n_degrees = []
#graph.degree, dann iterieren und durchschnitt nehmen
for graph in all_graphs_nx:
    degree_ary = graph.degree
    for degree_tuple in degree_ary:
        all_n_degrees.append(degree_tuple[1])
print('Average node degree: ', np.mean(all_n_degrees), ', std: ', np.std(all_n_degrees), ', sem: ', sem(all_n_degrees))

# average clust. coeff
all_avg_clust = []
for graph in all_graphs_nx:
    all_avg_clust.append(average_clustering(graph))
print('Average Clust. Coeff. :', np.mean(all_avg_clust), ', std: ', np.std(all_avg_clust), ', sem: ', sem(all_avg_clust))

#graph diameter
all_diameter = []
for graph in all_graphs_nx:
    try:
        all_diameter.append(diameter(graph))
    except networkx.exception.NetworkXError:
        continue
print('Average diameter :', np.mean(all_diameter), ', std: ', np.std(all_diameter), ', sem: ', sem(all_diameter))

