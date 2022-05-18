import matplotlib.pyplot as plt

from NodeAvg import NodeAvg
from G2VClust import G2VClust
from graph_helper.DatasetLoader import DatasetLoader
from sklearn.metrics import normalized_mutual_info_score as nmi, accuracy_score as acc, balanced_accuracy_score as bacc, adjusted_rand_score as ari, adjusted_mutual_info_score as ami, homogeneity_score as hmg
import numpy as np
import umap
from ceilo_graph_generation.grakeltest import silhouette
import os
from pathlib import Path
import json



datasets = ['ogbg-molhiv', 'TWITTER-Real-Graph-Partial', 'MNISTSuperpixels', 'ConstructedGraphs'] #'MNISTSuperpixels500', 'ogbg-ppa', 'ogbg-ppa10000',
classes = [2, 2, 10, 10, 37, 37, 2]
baselines = ['G2VClust', 'NodeAvg']
dataset_idx = 6
g2v_dim = 32
ds_string = datasets[dataset_idx]
dl = DatasetLoader()
x, y = dl.load_dataset_nx(ds_string)
x_p, _ = dl.load_dataset_ptg(ds_string)
Path(datasets[dataset_idx]).mkdir(exist_ok=True)
print('dataset loaded')



def get_embs_and_preds(baseline_idx):
    if baseline_idx==0:
        g2vc = G2VClust(g2v_dim)
        all_embs, y_pred = g2vc.kmeans_clust(classes[dataset_idx], ds_string)
    elif baseline_idx==1:
        na = NodeAvg()
        all_embs, y_pred = na.kmeans_clust(2, ds_string)
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


def save_stats_dict(y, y_pred):
    n_nodes_ary = [g.number_of_nodes() for g in x]
    n_edges_ary = [g.number_of_edges() for g in x]

    stats_dict = {'nmi':nmi(y, y_pred), 'ari':ari(y, y_pred), 'ami':ami(y, y_pred), 'hmg':hmg(y, y_pred),
                  'sil1':silhouette(x_p, y_pred, n_iter=1), 'sil2':silhouette(x_p, y_pred, n_iter=2),
                  'sil3':silhouette(x_p, y_pred, n_iter=3), 'sil5':silhouette(x_p, y_pred, n_iter=5),
                  'sil10':silhouette(x_p, y_pred, n_iter=10),
                  'mean_std_nodes':mean_std_per_group(n_nodes_ary, y_pred, classes[dataset_idx]),
                  'mean_std_edges':mean_std_per_group(n_edges_ary, y_pred, classes[dataset_idx])}

    with open(os.path.join(datasets[dataset_idx],'stats_dict.json'), 'w') as fp:
        json.dump(dict, fp)


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
        ax.set_title('{}, {}, Class {}'.format(datasets[dataset_idx], baselines[baseline_idx], str(class_i)))
        plt.savefig(os.path.join(datasets[dataset_idx],str(class_i)+'.png'))

for baseline_idx in range(len(baselines)):
    for dataset_idx in range(len(datasets)):
        x, y = dl.load_dataset_nx(datasets[dataset_idx])
        all_embs, y_pred = get_embs_and_preds(baseline_idx)
        stats_dict = save_stats_dict(y, y_pred)
        umap_emb = generate_umap_emb(all_embs)
        viz_classes(umap_emb)
