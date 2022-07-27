import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from modules.GraphTransform_All import GraphTransform_All
from modules.GraphTransform_All_TUD import GraphTransform_All_TUD
from modules.GraphTransform_All_molhiv import GraphTransform_All_molhiv
from modules.GraphTransform_NoNodeMask import GraphTransform_NoNodeMask
from modules.gcn import GCN
from utils import yaml_config_hook, save_model
from torch.utils import data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from modules.graph_utils import max_degree_undirected
import torch_geometric
import random
from copy import deepcopy
from torch.utils.data import ConcatDataset
from torch_geometric.datasets import MNISTSuperpixels, GNNBenchmarkDataset,  TUDataset
import time
from datasets.CeiloGraphs import CeiloGraphs
from ogb.graphproppred import PygGraphPropPredDataset
from graph_helper.DatasetLoader import DatasetLoader
from pathlib import Path
import multiprocessing as mp

dataset_string = "constructedgraphs_2"

dl = DatasetLoader()
dataset, y_p = dl.load_dataset_ptg(dataset_string)
GraphTransformer = GraphTransform_All(drop_nodes_ratio=0.2, subgraph_ratio=0.8, permute_edges_ratio=0.2,
                                      mask_nodes_ratio=0.2, batch_size=32)

_, _, data_i_na, data_j_na = GraphTransformer.generate_augmentations_i_j_noaug(dataset)
_, _, data_i_sg, data_j_sg = GraphTransformer.generate_augmentations_i_j_subgraphonly(dataset)
_, _, data_i_pe, data_j_pe = GraphTransformer.generate_augmentations_i_j_permuteedgesonly(dataset)
_, _, data_i_dn, data_j_dn = GraphTransformer.generate_augmentations_i_j_dropnodesonly(dataset)
_, _, data_i_mn, data_j_mn = GraphTransformer.generate_augmentations_i_j_masknodesonly(dataset)

print('NA')
print(data_i_na[0])
print(data_i_na[0].x)
print(data_i_na[0].edge_index)
print(data_j_na[0])
print('SG')
print(data_i_sg[0])
print(data_i_sg[0].x)
print(data_i_sg[0].edge_index)
print(data_j_sg[0])
print('PE')
print(data_i_pe[0])
print(data_i_pe[0].x)
print(data_i_pe[0].edge_index)
print(data_j_pe[0])
print('DN')
print(data_i_dn[0])
print(data_i_dn[0].x)
print(data_i_dn[0].edge_index)
print(data_j_dn[0])
print('MN')
print(data_i_mn[0])
print(data_i_mn[0].x)
print(data_i_mn[0].edge_index)
print(data_j_mn[0])