import math
import os
import itertools
import json
import pickle

import matplotlib.pyplot as plt
import networkx
import torch
import torch_geometric
import numpy as np
import networkx as nx

path_meta_labeled = "/media/rigel/Data/DWD/snrfiltered_samples_slicsegmented_labeled_{}_{}_{}_meta_heightcut".format(str(50),
                                                                                                   str(20),
                                                                                                   str(1))

max_dist = 50 #pixels
x_norm = 1440
y_norm = 256

meta_files = os.listdir(path_meta_labeled)

donelist=[]

def get_base_name(meta_file):
    current = meta_file[:-6]
    checklastflag = True
    while checklastflag:
        if current[-1].isdigit():
            current = current[:-1]
        else:
            checklastflag = False
    return current

def get_all_segmentfiles(base_name):
    all_files_lists = []
    for meta_file in meta_files:
        if get_base_name(meta_file)==base_name:
            all_files_lists.append(meta_file)
    return all_files_lists

def check_for_edge(seg1_coords, seg2_coords, max_dist):
    #seg1_coords = [[x1_min, x1_med, x1_max], [y1_min, y1_med, y1_max]]
    seg1_points = [i for i in itertools.product(seg1_coords[0], seg1_coords[1])] #[(x1_min, y1_min), (x1_min, y1_med),...]
    seg2_points = [i for i in itertools.product(seg2_coords[0], seg2_coords[1])]
    for seg1_point in seg1_points:
        for seg2_point in seg2_points:
            if math.dist(seg1_point, seg2_point)<max_dist:
                return True
    return False

def get_seg_coords(segmentfile):
    with open(os.path.join(path_meta_labeled, segmentfile)) as json_file:
        meta_dict = json.load(json_file)
    return [[meta_dict['x_min'], (meta_dict['x_min']+meta_dict['x_max'])/2, meta_dict['x_max']],
            [meta_dict['y_min'], (meta_dict['y_min']+meta_dict['y_max'])/2, meta_dict['y_max']]]

def get_node_features(all_segmentfiles):
    node_x = []
    for idx, segmentfile in enumerate(all_segmentfiles):
        with open(os.path.join(path_meta_labeled, segmentfile)) as json_file:
            meta_dict = json.load(json_file)
        features = [*meta_dict['label'], meta_dict['x_min']/x_norm, meta_dict['x_max']/x_norm,
                                            meta_dict['y_min']/y_norm, meta_dict['y_max']/y_norm]
        node_x.append(features)
    return node_x


all_graphs = {}
n=0
for meta_file in meta_files:
    n+=1
    print(n)
    base_name = get_base_name(meta_file)
    if base_name in donelist:
        continue
    all_segmentfiles = get_all_segmentfiles(base_name)
    node_x = get_node_features(all_segmentfiles)
    edge_index = [[],[]]
    for idx1, segmentfile1 in enumerate(all_segmentfiles):
        for idx2, segmentfile2 in enumerate(all_segmentfiles):
            if segmentfile1 != segmentfile2:
                if check_for_edge(get_seg_coords(segmentfile1), get_seg_coords(segmentfile2), max_dist):
                    #undirected graph
                    edge_index[0].append(idx1)
                    edge_index[1].append(idx2)
                    edge_index[0].append(idx2)
                    edge_index[1].append(idx1)
    donelist.append(base_name)
    edge_index_t = torch.tensor(np.array(edge_index), dtype=torch.long)
    x = torch.tensor(np.array(node_x), dtype=torch.float)

    data = torch_geometric.data.Data(x=x, edge_index=edge_index_t)
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)

    subgraphs_nx = [g.subgraph(c).copy() for c in nx.connected_components(g)]
    n_graphs=len(subgraphs_nx)

    '''
    for g in subgraphs:
        nx.draw(g)
        plt.show()
    '''
    subgraphs_t = [torch_geometric.utils.from_networkx(g) for g in subgraphs_nx]
    startnode=0
    subgraphs_final = []
    for subgraph in subgraphs_t:
        new_x=torch.tensor(np.array(node_x[startnode:subgraph.num_nodes+startnode]), dtype=torch.float)
        startnode+=subgraph.num_nodes
        subgraphs_final.append(torch_geometric.data.Data(x=new_x, edge_index=subgraph.edge_index))

    all_graphs[base_name]=subgraphs_final

with open('graphs_processed.pickle', 'wb') as handle:
    pickle.dump(all_graphs, handle)