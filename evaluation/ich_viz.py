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

def load_model(model_fp, model):
    checkpoint = torch.load(model_fp)
    model.load_state_dict(checkpoint['net'])
    return model

parser = argparse.ArgumentParser()
config = yaml_config_hook("../config/config.yaml")
for k, v in config.items():
    parser.add_argument(f"--{k}", default=v, type=type(v))
args = parser.parse_args()

if args.dataset == "MNISTSuperpixels":
    dataset = MNISTSuperpixels(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', train=True)
    '''
    dataset_train = MNISTSuperpixels(root='/home/md/PycharmProjects/CC4Graphs/datasets/', train=True)
    dataset_test = MNISTSuperpixels(root='/home/md/PycharmProjects/CC4Graphs/datasets/', train=False)
    dataset = ConcatDataset([dataset_train, dataset_test])
    '''
else:
    dataset_pretransform = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name=args.dataset)
    dataset = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name=args.dataset, transform=torch_geometric.transforms.OneHotDegree(max_degree=max_degree_undirected(dataset_pretransform)))
class_num = dataset.num_classes
print('Dataset loaded')

gnn = GCN(dataset.num_features)
model = network.Network(gnn, args.feature_dim, class_num)
model_fp = os.path.join("/home/rigel/MDammann/PycharmProjects/CC4Graphs/save/COLLAB_test", "checkpoint_30.tar".format(args.start_epoch))
model = load_model(model_fp, model)
print('Model loaded')

all_data = [elem for elem in dataset]
all_data = all_data[0:30000]
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
all_clusters_inverted=[]
for elem in all_clusters:
    if elem == 0:
        all_clusters_inverted.append(1)
    else:
        all_clusters_inverted.append(0)
all_clusters_inverted=np.array(all_clusters_inverted)
print(all_clusters)
print(all_clusters_inverted)
print(nmi(all_y, all_clusters))
print(accuracy_score(all_y, all_clusters))

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


