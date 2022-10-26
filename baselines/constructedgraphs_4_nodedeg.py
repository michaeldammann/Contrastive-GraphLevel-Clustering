import random
from random import randrange, sample
import pickle
#graph1: 5-10 nodes, features [1,1,1,0,0,0], knotengrad 1-3
#graph2: 15-20 nodes, features [0,0,0,1,1,1], knotengrad 6-8
#zuerst path für zusammenhang, dann hinzufügen

random.seed(42)

graph1_min_max_nodes = (5,11)
graph1_min_max_nodedeg = (1,4)
graph2_min_max_nodes = (15,26)
graph2_min_max_nodedeg = (5,8)

def node_feat_choice(pref):
    if pref:
        return random.choices([0,1], [0.5, 0.5])[0]
    else:
        return random.choices([0, 1], [0.8, 0.2])[0]

def node_feats(graph_class):
    if graph_class == 0 or graph_class == 3:
        return [node_feat_choice(True), node_feat_choice(True), node_feat_choice(True), node_feat_choice(False),node_feat_choice(False),node_feat_choice(False)]
    elif graph_class == 1 or graph_class == 2:
        return [node_feat_choice(False), node_feat_choice(False), node_feat_choice(False), node_feat_choice(True), node_feat_choice(True), node_feat_choice(True)]

def not_neighbors(x,edge_index, node_idx):
    all_node_idcs =  [i for i in range(len(x))]
    all_node_idcs.remove(node_idx)
    for i in range(len(edge_index[0])):
        if edge_index[0][i] == node_idx:
            all_node_idcs.remove(edge_index[1][i])
    return all_node_idcs

def generate_graph(graph_class):
    x=[]
    edge_index=[[],[]]

    if graph_class == 0 or graph_class == 3:
        n_nodes = randrange(*graph1_min_max_nodes)
    elif graph_class == 1 or graph_class == 2:
        n_nodes = randrange(*graph2_min_max_nodes)

    #init node features
    for i in range(n_nodes):
        x.append(node_feats(graph_class))

    #path through graph
    for i in range(n_nodes-1):
        edge_index[0].append(i)
        edge_index[1].append(i+1)
        edge_index[0].append(i+1)
        edge_index[1].append(i)

    for i in range(n_nodes):
        if graph_class == 0 or graph_class == 2:
            nodedeg = randrange(*graph1_min_max_nodedeg)
        elif graph_class == 1 or graph_class == 3:
            nodedeg = randrange(*graph2_min_max_nodedeg)

        not_neighs=not_neighbors(x, edge_index,i)
        for e in range(min(nodedeg, len(not_neighs))):
            new_neigh = sample(not_neighs, 1)[0]
            not_neighs.remove(new_neigh)
            edge_index[0].append(i)
            edge_index[1].append(new_neigh)
            edge_index[0].append(new_neigh)
            edge_index[1].append(i)

    return (x, edge_index, graph_class)

all_graphs = []
for i in range(30000):
    all_graphs.append(generate_graph(0))
for i in range(30000):
    all_graphs.append(generate_graph(1))
for i in range(30000):
    all_graphs.append(generate_graph(2))
for i in range(30000):
    all_graphs.append(generate_graph(3))

with open('constructedgraphs_4nodedeg.pkl', 'wb') as handle:
    pickle.dump(all_graphs, handle)


