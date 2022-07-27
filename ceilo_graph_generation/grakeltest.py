from grakel import Graph
from grakel.kernels import MultiscaleLaplacian, ShortestPath, WeisfeilerLehman, VertexHistogram
from grakel import GraphKernel
from graph_helper.DatasetLoader import DatasetLoader
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import silhouette_score
import numpy as np
import random



def ptg_to_grakel(ptg_graph):
    adj_matrix = to_dense_adj(ptg_graph.edge_index).cpu().detach().numpy()[0].tolist()
    nodelist = [i for i in range(ptg_graph.x.cpu().detach().numpy().shape[0])]
    featurelist = [tuple(f) for f in ptg_graph.x.cpu().detach().numpy().tolist()]
    node_to_features = dict(zip(nodelist, featurelist))
    return Graph(initialization_object=adj_matrix, node_labels=node_to_features)

def silhouette(dataset_x, labels, n_iter=5, samples=2500):
    random.seed(42)
    wl_kernel = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram, normalize=True)
    dataset_gk=[ptg_to_grakel(g) for g in dataset_x]
    dataset_gk_s, labels_s = zip(*random.sample(list(zip(dataset_gk, labels)), samples))
    wlmatrix=wl_kernel.fit_transform(dataset_gk_s)
    distance = (1.-np.array(wlmatrix)).tolist()
    return silhouette_score(X=distance, labels=labels_s, metric='precomputed')

'''
dl = DatasetLoader()
x, y = dl.load_dataset_ptg('ogbg-molhiv')
ptg_to_grakel(x[0])


adj = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]
node_attributes = {0: [1.2, 0.5], 1: [2.8, -0.6], 2: [0.7, 1.1]}
G = Graph(adj, node_labels=node_attributes)


from grakel import Graph

H2O_adjacency = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]

H2O_node_labels = {0: tuple([0,1]), 1: tuple([1,0]), 2: tuple([1,0])}
#H2O_node_labels = {0: 0, 1: 1, 2: 2}

H2O = Graph(initialization_object=H2O_adjacency, node_labels=H2O_node_labels)

H3O_adjacency = [[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]

H3O_node_labels = {0: tuple([0,1]), 1: tuple([1,0]), 2: tuple([1,0]), 3:tuple([1,0])}
#H3O_node_labels = {0: 0, 1: 1, 2: 2, 3:3}

H3O = Graph(initialization_object=H3O_adjacency, node_labels=H3O_node_labels)

sp_kernel = ShortestPath()
ml_kernel = MultiscaleLaplacian()
wl_kernel = WeisfeilerLehman(n_iter=1, base_graph_kernel=VertexHistogram, normalize=True)

print(wl_kernel.fit_transform([ptg_to_grakel(x[0]), ptg_to_grakel(x[1])]))
#print(wl_kernel.transform([H3O]))
#print(wl_kernel.transform([H2O]))


###########simple test##################

from grakel import Graph
g1_adjacency = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]

g1_node_labels = {0: tuple([0,1]), 1: tuple([1,0]), 2: tuple([1,0])}
#H2O_node_labels = {0: 0, 1: 1, 2: 2}

g1 = Graph(initialization_object=g1_adjacency, node_labels=g1_node_labels)

g2_adjacency = [[0, 1, 1], [1, 0, 0], [1, 0, 0]]

g2_node_labels = {0: tuple([1,0]), 1: tuple([1,0]), 2: tuple([1,0])}
#H2O_node_labels = {0: 0, 1: 1, 2: 2}

g2 = Graph(initialization_object=g2_adjacency, node_labels=g2_node_labels)

wlg = GraphKernel(kernel="weisfeiler_lehman")

wl_kernel = WeisfeilerLehman(n_iter=1, base_graph_kernel=VertexHistogram, normalize=True)
print(wlg.fit_transform([g1, g2]))
print(wlg.fit_transform([g1, g2])
'''
'''
'''