import matplotlib.pyplot as plt

from NodeAvg import NodeAvg
from G2VClust import G2VClust
from FGSDClust import FGSDClust
from SFClust import SFClust
from GeoScatteringClust import GeoScatteringClust
from graph_helper.DatasetLoader import DatasetLoader
from sklearn.metrics import normalized_mutual_info_score as nmi, accuracy_score as acc, \
    balanced_accuracy_score as bacc, adjusted_rand_score as ari, adjusted_mutual_info_score as ami, \
    homogeneity_score as hmg, fowlkes_mallows_score as fm
import numpy as np
import umap
from ceilo_graph_generation.grakeltest import silhouette
import os
from pathlib import Path
import json
import random
from modules.gcn import GCN
from modules import network
from torch_geometric.loader import DataLoader
import torch
from adjust_labels import get_y_preds

np.random.seed(42)
random.seed(42)



datasets = ['ogbg-molhiv', 'TWITTER-Real-Graph-Partial', 'MNISTSuperpixels', 'constructedgraphs_2', 'constructedgraphs_2nodedeg', 'constructedgraphs_4', 'reddit_threads', 'twitch_egos'] #'MNISTSuperpixels500', 'ogbg-ppa', 'ogbg-ppa10000',
classes = [2, 2, 10, 2, 2, 4, 2, 2]#, 37, 37, 2]

models = ['G2VClust', 'NodeAvg', 'CC', 'FGSDClust', 'SFClust', 'GeoScatteringClust']
#dataset_idx = 6
g2v_dim = 32
dl = DatasetLoader()

cc_basepath = '/home/rigel/MDammann/PycharmProjects/CC4Graphs/save/'
dataset_dicts_cc = [{'num_features':9, 'start_epoch': 60, 'avg_num_nodes':25},
                    {'num_features':1323, 'start_epoch': 50, 'avg_num_nodes':4},
                    {'num_features':1, 'start_epoch':500, 'avg_num_nodes':75},
                    {'num_features':6, 'start_epoch':290, 'avg_num_nodes':13},
                    {},
                    {'num_features':6, 'start_epoch':70},
                    {'num_features':94, 'start_epoch':280, 'avg_num_nodes':23},
                    {'num_features':52, 'start_epoch':280, 'avg_num_nodes':29}]
cc_dict = {'feature_dim':64}
'''
ds_string = datasets[dataset_idx]

x, y = dl.load_dataset_nx(ds_string)
x_p, _ = dl.load_dataset_ptg(ds_string)
Path(datasets[dataset_idx]).mkdir(exist_ok=True)
print('dataset loaded')
'''

def load_model(model_fp, model):
    checkpoint = torch.load(model_fp)
    model.load_state_dict(checkpoint['net'])
    return model


def get_embs_and_preds(models_idx, x_p):
    if models_idx==0:
        g2vc = G2VClust(g2v_dim)
        all_embs, y_pred = g2vc.kmeans_clust(classes[dataset_idx], datasets[dataset_idx])
    elif models_idx==1:
        na = NodeAvg()
        all_embs, y_pred = na.kmeans_clust(2, datasets[dataset_idx])
    elif model_idx==2:
        gnn = GCN(dataset_dicts_cc[dataset_idx]['num_features'], dataset_dicts_cc[dataset_idx]['avg_num_nodes'])
        model = network.Network(gnn, cc_dict['feature_dim'], classes[dataset_idx])
        model_fp = os.path.join(cc_basepath, datasets[dataset_idx],
                                "checkpoint_{}.tar".format(dataset_dicts_cc[dataset_idx]['start_epoch']))
        model = load_model(model_fp, model)
        print('Model loaded')

        data_loader = DataLoader(x_p, batch_size=16)
        data_loader = iter(data_loader)
        print('Dataloader initialized')

        all_embs= []
        y_pred = []


        for step in range(len(data_loader)):
            x_i = next(data_loader)
            x_j = x_i
            z_i, z_j, c_i, c_j = model(x_i, x_j)
            z_i_np = z_i.cpu().detach().numpy()
            c_i_np = c_i.cpu().detach().numpy()
            all_embs.extend(z_i_np)
            y_pred.extend(np.argmax(c_i_np, axis=1))
    elif models_idx==3:
        fgsdc = FGSDClust(g2v_dim)
        all_embs, y_pred = fgsdc.kmeans_clust(classes[dataset_idx], datasets[dataset_idx])
    elif models_idx==4:
        sfc = SFClust(g2v_dim)
        all_embs, y_pred = sfc.kmeans_clust(classes[dataset_idx], datasets[dataset_idx])
    elif models_idx==5:
        gsc = GeoScatteringClust(g2v_dim)
        all_embs, y_pred = gsc.kmeans_clust(classes[dataset_idx], datasets[dataset_idx])

    return all_embs, y_pred

def mean_std_per_group(ary, group_ary, n_classes):
    n_nodes_grouped = []
    for i in range(n_classes):
        i_nodes = []
        for n_nodes_idx, n_nodes in enumerate(ary):
            if group_ary[n_nodes_idx]==i:
                i_nodes.append(n_nodes)
        n_nodes_grouped.append(i_nodes)
    mean_ary = []
    for i in range(n_classes):
        mean_ary.append(np.mean(n_nodes_grouped[i]))
    std_ary = []
    for i in range(n_classes):
        std_ary.append(np.std(n_nodes_grouped[i]))
    return mean_ary, std_ary


def save_stats_dict(x_p, y, y_pred):
    n_nodes_ary = [g.number_of_nodes() for g in x]
    n_edges_ary = [g.number_of_edges() for g in x]

    pred_adjusted = get_y_preds(y, y_pred, len(set(y)))

    stats_dict = {'nmi':nmi(y, y_pred), 'ari':ari(y, y_pred), 'ami':ami(y, y_pred), 'hmg':hmg(y, y_pred),
                  'acc': acc(y, pred_adjusted), 'bacc':bacc(y, pred_adjusted), 'fm': fm(y, pred_adjusted),
                  'sil1':silhouette(x_p, y_pred, n_iter=1), 'sil2':silhouette(x_p, y_pred, n_iter=2),
                  'sil3':silhouette(x_p, y_pred, n_iter=3), 'sil5':silhouette(x_p, y_pred, n_iter=5),
                  'sil10':silhouette(x_p, y_pred, n_iter=10),
                  'mean_std_nodes':mean_std_per_group(n_nodes_ary, y_pred, classes[dataset_idx]),
                  'mean_std_edges':mean_std_per_group(n_edges_ary, y_pred, classes[dataset_idx])}

    with open(os.path.join('{}_{}'.format(datasets[dataset_idx],models[model_idx]),'stats_dict.json'), 'w') as fp:
        json.dump(stats_dict, fp)


def generate_umap_emb(all_embs):
    reducer = umap.UMAP(random_state=42)
    reducer.fit(all_embs)
    return reducer.transform(all_embs)


def viz_classes(umap_embedding):
    cdict_bin = {1:'red', 0:'grey'}

    for class_i in range(classes[dataset_idx]):
        plt.clf()
        y_bin = [1 if y_i == class_i else 0 for y_i in y]
        fig, ax = plt.subplots()
        ax.set_xlabel('UMAP Dim. 1')
        ax.set_ylabel('UMAP Dim. 2')
        for g in np.unique(y_bin):
            ix = np.where(y_bin == g)
            ax.scatter(umap_embedding[ix,0], umap_embedding[ix,1], c = cdict_bin[g], label = g, alpha = 0.2)
        ax.set_title('{}, {}, Class {}'.format(datasets[dataset_idx], models[model_idx], str(class_i)))
        plt.savefig(os.path.join('{}_{}'.format(datasets[dataset_idx],models[model_idx]),str(class_i)+'.png'))

for model_idx in [2]:#range(len(models)):
    print('Model:', models[model_idx])
    for dataset_idx in range(6,7):#range(len(datasets)):
        Path('{}_{}'.format(datasets[dataset_idx],models[model_idx])).mkdir(exist_ok=True)
        print('Dataset:', datasets[dataset_idx])
        x, y = dl.load_dataset_nx(datasets[dataset_idx])
        x_p, y_p = dl.load_dataset_ptg(datasets[dataset_idx])
        all_embs, y_pred = get_embs_and_preds(model_idx, x_p)
        stats_dict = save_stats_dict(x_p, y, y_pred)
        umap_emb = generate_umap_emb(all_embs)
        viz_classes(umap_emb)
