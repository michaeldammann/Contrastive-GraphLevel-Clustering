from modules.graph_transform import drop_nodes, permute_edges, subgraph, mask_nodes
from torch_geometric.datasets import TUDataset
from copy import deepcopy
import numpy as np
import random
from sys import getsizeof

drop_nodes_ratio, subgraph_ratio, permute_edges_ratio, mask_nodes_ratio = 0.2, 0.2, 0.2, 0.2

dataset = TUDataset(root='/home/md/PycharmProjects/CC4Graphs/datasets/', name='NCI1')

all_data = [elem for elem in dataset]

random.shuffle(all_data)

all_data_i, all_data_j = deepcopy(all_data), deepcopy(all_data)

print(all_data_i[0])
print(all_data_j[0])
print(all_data_i[1])
print(all_data_j[1])

for idx, elem in enumerate(all_data):
    ri, rj = np.random.randint(5), np.random.randint(5)
    if ri == 0:
        drop_nodes(all_data_i[idx], drop_nodes_ratio)
    elif ri == 1:
        subgraph(all_data_i[idx], subgraph_ratio)
    elif ri == 2:
        permute_edges(all_data_i[idx], permute_edges_ratio)
    elif ri == 3:
        mask_nodes(all_data_i[idx], mask_nodes_ratio)
    #elif ri == 4: identity

    if rj == 0:
        drop_nodes(all_data_j[idx], drop_nodes_ratio)
    elif rj == 1:
        subgraph(all_data_j[idx], subgraph_ratio)
    elif rj == 2:
        permute_edges(all_data_j[idx], permute_edges_ratio)
    elif rj == 3:
        mask_nodes(all_data_j[idx], mask_nodes_ratio)
    #elif rj == 4: identity

print(all_data_i[0])
print(all_data_j[0])
print(all_data_i[1])
print(all_data_j[1])