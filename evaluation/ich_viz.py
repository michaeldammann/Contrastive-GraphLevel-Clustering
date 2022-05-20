import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from modules.GraphTransform_All import GraphTransform_All
from modules.gcn import GCN
from modules.graph_utils import max_degree_undirected
from utils import yaml_config_hook, save_model
from torch.utils.data import ConcatDataset
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.loader import DataLoader
import random
import umap
from matplotlib import pyplot as plt
from datasets.CeiloGraphs import CeiloGraphs
from ogb.graphproppred import PygGraphPropPredDataset
from graph_helper.DatasetLoader import DatasetLoader
from evaluation import get_y_preds

def load_model(model_fp, model):
    checkpoint = torch.load(model_fp)
    model.load_state_dict(checkpoint['net'])
    return model

parser = argparse.ArgumentParser()
config = yaml_config_hook("../config/config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))
args = parser.parse_args()

if args.dataset == "CeiloGraphs":
    dataset = CeiloGraphs(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/',
                          graphpicklepath=args.graphpicklepath)
    class_num = args.ceilo_n_clusters
    num_features = dataset.num_features
elif args.dataset == 'ogbg-molhiv':
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
    class_num = 2
    num_features = dataset.num_features
elif args.dataset == 'TWITTER-Real-Graph-Partial':
    dataset = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name=args.dataset)
    class_num = dataset.num_classes
    num_features = dataset.num_features
elif args.dataset == "MNISTSuperpixels":
    dataset = MNISTSuperpixels(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/MNISTSuperpixels',
                               train=True)
    class_num = dataset.num_classes
    num_features = dataset.num_features
elif args.dataset in ["constructedgraphs_2", "constructedgraphs_4", "constructedgraphs_2nodedeg"]:
    dl = DatasetLoader()
    dataset, y_p = dl.load_dataset_ptg(args.dataset)
    GraphTransformer = GraphTransform_All(drop_nodes_ratio=0.2, subgraph_ratio=0.8, permute_edges_ratio=0.2,
                                          mask_nodes_ratio=0.2, batch_size=args.batch_size)
    class_num = len(np.unique(y_p))
    num_features = len(dataset[0].x[0].cpu().detach().numpy())
    print(class_num, num_features)
else:
    dataset_pretransform = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name=args.dataset)
    dataset = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name=args.dataset,
                        transform=torch_geometric.transforms.OneHotDegree(
                            max_degree=max_degree_undirected(dataset_pretransform)))
    class_num = dataset.num_classes
    num_features = dataset.num_features
print('Dataset loaded')

gnn = GCN(num_features)
model = network.Network(gnn, args.feature_dim, class_num)
model_fp = os.path.join("/home/rigel/MDammann/PycharmProjects/CC4Graphs/save/constructedgraphs_2", "checkpoint_250.tar".format(args.start_epoch))
model = load_model(model_fp, model)
print('Model loaded')

all_data = [elem for elem in dataset]
#all_data = all_data[0:30000]
data_loader=DataLoader(all_data, batch_size=16)
data_loader=iter(data_loader)
print('Dataloader initialized')

all_z_i_np = []
all_clusters = []
all_y = []

for step in range(len(data_loader)):
    x_i = next(data_loader)
    y_np = x_i.y.cpu().detach().numpy()
    x_j = x_i
    z_i, z_j, c_i, c_j = model(x_i, x_j)
    z_i_np = z_i.cpu().detach().numpy()
    c_i_np = c_i.cpu().detach().numpy()
    all_z_i_np.extend(z_i_np)
    all_clusters.extend(np.argmax(c_i_np, axis=1))
    all_y.extend(y_np)

all_z_i_np=np.array(all_z_i_np)
all_clusters=np.array(all_clusters)
all_y=np.array(all_y)

print(all_z_i_np.shape)
print(all_clusters.shape)
print(all_y.shape)
print(all_clusters[0:100])
print(all_y[0:100])

from sklearn.metrics import normalized_mutual_info_score as nmi, accuracy_score
print("NMI")

print(nmi(all_y, all_clusters))
all_clusters_adjusted = get_y_preds(all_y, all_clusters, len(set(all_y)))
print(accuracy_score(all_clusters_adjusted, all_y))

reducer = umap.UMAP()
embedding = reducer.fit_transform(all_z_i_np)


scatter_x = embedding[:, 0]
scatter_y = embedding[:, 1]
group = all_y
cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'magenta', 4: 'grey', 5: 'black', 6:'yellow', 7:'orange', 8:'darkgoldenrod', 9:'darkviolet'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 30, alpha=0.1)
ax.legend()
plt.show()


