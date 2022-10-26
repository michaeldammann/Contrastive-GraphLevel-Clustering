import matplotlib.pyplot as plt

from NodeAvg import NodeAvg
from G2VClust import G2VClust
from FGSDClust import FGSDClust
from SFClust import SFClust
from GL2VClust import GL2VClust
from SFGL2VClust import SFGL2VClust
from GeoScatteringClust import GeoScatteringClust
from graph_helper.DatasetLoader import DatasetLoader
from sklearn.metrics import normalized_mutual_info_score as nmi, accuracy_score as acc, \
    balanced_accuracy_score as bacc, adjusted_rand_score as ari, adjusted_mutual_info_score as ami, \
    homogeneity_score as hmg, fowlkes_mallows_score as fm
import numpy as np
import umap
from ceilo_graph_generation.grakeltest import silhouette
from modules.gcnsimplepool import GCNSimplePool
import os
from pathlib import Path
import json
import random
from modules.gcn import GCN
from modules import network
from torch_geometric.loader import DataLoader
import torch
from adjust_labels import get_y_preds
from sklearn.cluster import KMeans
import pickle

np.random.seed(42)
random.seed(42)

'''
datasets = ['ceilographs4_features']
'''
datasets = ['ceilographs4_year', 'ceilographs4_month', 'ceilographs4_location', 'ceilographs4_season', 'ceilographs4_nodenumber', 'ceilographs4_edgenumber', 'ceilographs4_features',
            'ceilographs8_year', 'ceilographs8_month', 'ceilographs8_location', 'ceilographs8_season', 'ceilographs8_nodenumber', 'ceilographs8_edgenumber', 'ceilographs8_features',
            'ceilographs16_year', 'ceilographs16_month', 'ceilographs16_location', 'ceilographs16_season', 'ceilographs16_nodenumber', 'ceilographs16_edgenumber', 'ceilographs16_features']


classes = [4, 4, 4, 4, 4, 4, 4,
           8, 8, 8, 8, 8, 8, 8,
           16, 16, 16, 16, 16, 16, 16]
augmodes = ["default", "dropnodesonly", "masknodesonly", "noaug", "permuteedgesonly", "subgraphonly", "default_clusteronly", "default_instanceonly", "subgraphanddropnodesonly", "default_gcnsimplepool"]

models = ['G2VClust', 'NodeAvg', 'CC', 'FGSDClust', 'SFClust', 'GeoScatteringClust', 'GL2VClust', 'CCInstanceOnly', 'SFGL2VClust']
#dataset_idx = 6
g2v_dim = 64
dl = DatasetLoader()

cc_basepath = '/home/rigel/MDammann/PycharmProjects/CC4Graphs/save/'
dataset_dicts_cc = [{'num_features':16, 'start_epoch': [500], 'avg_num_nodes':8} for i in range(len(datasets))]
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


def get_embs_and_preds(models_idx, x_p, aug_idx):
    if models_idx==0:
        g2vc = G2VClust(g2v_dim)
        all_embs, y_pred = g2vc.kmeans_clust(classes[dataset_idx], datasets[dataset_idx])
    elif models_idx==1:
        na = NodeAvg()
        all_embs, y_pred = na.kmeans_clust(2, datasets[dataset_idx])
    elif model_idx==2:
        if augmode == "default_gcnsimplepool":
            gnn = GCNSimplePool(dataset_dicts_cc[dataset_idx]['num_features'], avg_num_nodes=0)
        else:
            gnn = GCN(dataset_dicts_cc[dataset_idx]['num_features'], dataset_dicts_cc[dataset_idx]['avg_num_nodes'])
        #gnn = GCN(dataset_dicts_cc[dataset_idx]['num_features'], dataset_dicts_cc[dataset_idx]['avg_num_nodes'])
        model = network.Network(gnn, cc_dict['feature_dim'], classes[dataset_idx])
        model_fp = os.path.join(cc_basepath, "{}_{}".format(datasets[dataset_idx].split('_')[0], augmodes[aug_idx]),
                                "checkpoint_{}.tar".format(dataset_dicts_cc[dataset_idx]['start_epoch'][aug_idx]))
        print(model_fp)
        model = load_model(model_fp, model)
        print('Model loaded')

        data_loader = DataLoader(x_p, batch_size=16)
        data_loader = iter(data_loader)
        print('Dataloader initialized')

        all_embs= []
        y_pred = []
        y_probs = []
        y_probs_full = []


        for step in range(len(data_loader)):
            x_i = next(data_loader)
            x_j = x_i
            z_i, z_j, c_i, c_j = model(x_i, x_j)
            z_i_np = z_i.cpu().detach().numpy()
            c_i_np = c_i.cpu().detach().numpy()
            all_embs.extend(z_i_np)
            y_pred.extend(np.argmax(c_i_np, axis=1))
            y_probs.extend(np.amax(c_i_np, axis=1))
            y_probs_full.extend(c_i_np)
    elif models_idx==3:
        fgsdc = FGSDClust(g2v_dim)
        all_embs, y_pred = fgsdc.kmeans_clust(classes[dataset_idx], datasets[dataset_idx])
    elif models_idx==4:
        sfc = SFClust(g2v_dim)
        all_embs, y_pred = sfc.kmeans_clust(classes[dataset_idx], datasets[dataset_idx])
    elif models_idx==5:
        gsc = GeoScatteringClust(g2v_dim)
        all_embs, y_pred = gsc.kmeans_clust(classes[dataset_idx], datasets[dataset_idx])
    elif models_idx==6:
        gl2vc = GL2VClust(g2v_dim)
        all_embs, y_pred = gl2vc.kmeans_clust(classes[dataset_idx], datasets[dataset_idx])
    elif models_idx==7:
        gnn = GCN(dataset_dicts_cc[dataset_idx]['num_features'], dataset_dicts_cc[dataset_idx]['avg_num_nodes'])
        model = network.Network(gnn, cc_dict['feature_dim'], classes[dataset_idx])
        model_fp = os.path.join(cc_basepath, "{}_{}".format(datasets[dataset_idx], augmodes[aug_idx]),
                                "checkpoint_300.tar".format(dataset_dicts_cc[dataset_idx]['start_epoch'][aug_idx]))
        print(model_fp)
        model = load_model(model_fp, model)
        print('Model loaded')

        data_loader = DataLoader(x_p, batch_size=16)
        data_loader = iter(data_loader)
        print('Dataloader initialized')

        all_embs= []


        for step in range(len(data_loader)):
            x_i = next(data_loader)
            x_j = x_i
            z_i, z_j, c_i, c_j = model(x_i, x_j)
            z_i_np = z_i.cpu().detach().numpy()
            all_embs.extend(z_i_np)

        kmeans = KMeans(n_clusters=classes[dataset_idx], random_state=0).fit(all_embs)
        y_pred = kmeans.labels_
    elif models_idx==8:
        sfgl2vc = SFGL2VClust(g2v_dim)
        all_embs, y_pred = sfgl2vc.kmeans_clust(classes[dataset_idx], datasets[dataset_idx])

    return all_embs, y_pred, y_probs, y_probs_full

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


def save_stats_dict(x_p, y, y_pred, savedir):
    n_nodes_ary = [g.number_of_nodes() for g in x]
    n_edges_ary = [g.number_of_edges() for g in x]

    pred_adjusted = get_y_preds(y, y_pred, len(set(y)))

    stats_dict = {'nmi':nmi(y, y_pred), 'ari':ari(y, y_pred), 'ami':ami(y, y_pred), 'hmg':hmg(y, y_pred),
                  'acc': acc(y, pred_adjusted), 'bacc':bacc(y, pred_adjusted), 'fm': fm(y, pred_adjusted),
                  '''
                  'sil1':silhouette(x_p, y_pred, n_iter=1), 'sil2':silhouette(x_p, y_pred, n_iter=2),
                  'sil3':silhouette(x_p, y_pred, n_iter=3), 'sil5':silhouette(x_p, y_pred, n_iter=5),
                  'sil10':silhouette(x_p, y_pred, n_iter=10),
                  '''
                  'mean_std_nodes':mean_std_per_group(n_nodes_ary, y_pred, classes[dataset_idx]),
                  'mean_std_edges':mean_std_per_group(n_edges_ary, y_pred, classes[dataset_idx])}

    with open(os.path.join(savedir, 'stats_dict.json'), 'w') as fp:
    #with open(os.path.join('{}_{}'.format(datasets[dataset_idx],models[model_idx]),'stats_dict.json'), 'w') as fp:
        json.dump(stats_dict, fp)


def generate_umap_emb(all_embs, reducer):
    reducer.fit(all_embs)
    return reducer.transform(all_embs)


def viz_classes(umap_embedding, y, savedir):
    cdict_bin = {1:'red', 0:'grey'}

    for class_i in np.unique(y):#range(classes[dataset_idx]):
        plt.clf()
        y_bin = [1 if y_i == class_i else 0 for y_i in y]
        fig, ax = plt.subplots()
        ax.set_xlabel('UMAP Dim. 1')
        ax.set_ylabel('UMAP Dim. 2')
        for g in np.unique(y_bin):
            ix = np.where(y_bin == g)
            ax.scatter(umap_embedding[ix,0], umap_embedding[ix,1], c = cdict_bin[g], label = g, alpha = 0.2)
        ax.set_title('{}, {}, Class {}'.format(datasets[dataset_idx], models[model_idx], str(class_i)))
        plt.savefig(os.path.join(savedir, str(class_i) + '.png'))
        #plt.savefig(os.path.join('{}_{}'.format(datasets[dataset_idx],models[model_idx]),str(class_i)+'.png'))

def viz_classes_allinone(umap_embedding, y, savedir):
    cdict_bin = {0:'blue', 1:'red', 2:'green', 3:'orange', 4:'cyan', 5:'magenta', 6:'yellow', 7:'darkred', 8:'teal',
                 9:'darkviolet', 10:'chocolate', 11:'darkslategray', 12:'lime', 13:'darkgoldenrod', 14:'navajowhite',
                 15:'pink'}

    plt.clf()
    y_bin = y
    fig, ax = plt.subplots()
    ax.set_xlabel('UMAP Dim. 1')
    ax.set_ylabel('UMAP Dim. 2')
    for g in np.unique(y_bin):
        ix = np.where(y_bin == g)
        ax.scatter(umap_embedding[ix,0], umap_embedding[ix,1], c = cdict_bin[g], label = g, alpha = 0.03)
    ax.set_title('{}, {}'.format(datasets[dataset_idx], models[model_idx]))
    plt.savefig(os.path.join(savedir, 'allinone' + '.png'))
    #plt.savefig(os.path.join('{}_{}'.format(datasets[dataset_idx],models[model_idx]),str(class_i)+'.png'))

def viz_pred_classes(umap_embedding, y_pred, savedir):
    cdict_bin = {1:'red', 0:'grey'}

    for class_i in range(classes[dataset_idx]):
        plt.clf()
        y_bin = [1 if y_i == class_i else 0 for y_i in y_pred]
        fig, ax = plt.subplots()
        ax.set_xlabel('UMAP Dim. 1')
        ax.set_ylabel('UMAP Dim. 2')
        for g in np.unique(y_bin):
            ix = np.where(y_bin == g)
            ax.scatter(umap_embedding[ix,0], umap_embedding[ix,1], c = cdict_bin[g], label = g, alpha = 0.3)
        ax.set_title('{}, {}, Cluster {}'.format(datasets[dataset_idx], models[model_idx], str(class_i)))
        plt.savefig(os.path.join(savedir, 'cluster' + str(class_i) + '.png'))
        #plt.savefig(os.path.join('{}_{}'.format(datasets[dataset_idx],models[model_idx]),'cluster'+str(class_i)+'.png'))

def viz_pred_classes_allinone(umap_embedding, y_pred, savedir):
    cdict_bin = {0:'blue', 1:'red', 2:'green', 3:'orange', 4:'cyan', 5:'magenta', 6:'yellow', 7:'darkred', 8:'teal',
                 9:'darkviolet', 10:'chocolate', 11:'darkslategray', 12:'lime', 13:'darkgoldenrod', 14:'navajowhite',
                 15:'pink'}


    plt.clf()
    y_bin = y_pred
    fig, ax = plt.subplots()
    ax.set_xlabel('UMAP Dim. 1')
    ax.set_ylabel('UMAP Dim. 2')
    for g in np.unique(y_bin):
        ix = np.where(y_bin == g)
        ax.scatter(umap_embedding[ix,0], umap_embedding[ix,1], c = cdict_bin[g], label = g, alpha = 0.02)
    ax.set_title('{}, {}, Clusters'.format(datasets[dataset_idx], models[model_idx]))
    plt.savefig(os.path.join(savedir, 'cluster_allinone' + '.png'))
    #plt.savefig(os.path.join('{}_{}'.format(datasets[dataset_idx],models[model_idx]),'cluster'+str(class_i)+'.png'))

def sample_umap_embeddings(umap_embedding, x_p, y_p, y_pred, savedir, n_samples=100):
    all_samples = []
    samples = random.sample(range(len(x_p)), n_samples)
    for sample in samples:
        all_samples.append({"index":int(sample), "y_pred":int(y_pred[sample]), "umap_dims": [str(umap_embedding[sample,0]), str(umap_embedding[sample,1])], "num_nodes": str(x_p[sample].num_nodes), "num_edges": str(x_p[sample].num_edges), "features": str(np.sum(x_p[sample].x.cpu().detach().numpy(), axis=0)), "class": str(y_p[sample])})
    with open(os.path.join(savedir, 'umapsamples.json'), 'w') as f:
        json.dump(all_samples, f)
'''
############### Other than CC Evaluation ######################
for model_idx in [0,4,6]:# [0,4,6,8]:#range(len(models)):
    print('Model:', models[model_idx])
    reducer = umap.UMAP(random_state=42)
    for dataset_idx in [14]:#range(len(datasets)):
        Path('{}_{}'.format(datasets[dataset_idx], models[model_idx])).mkdir(exist_ok=True)
        print('Dataset:', datasets[dataset_idx])
        x, y = dl.load_dataset_nx(datasets[dataset_idx])
        x_p, y_p = dl.load_dataset_ptg(datasets[dataset_idx])

        Path('{}_{}'.format(datasets[dataset_idx], models[model_idx])).mkdir(exist_ok=True)
        all_embs, y_pred = get_embs_and_preds(model_idx, x_p, -1)
        savedir = Path('{}_{}'.format(datasets[dataset_idx], models[model_idx]))
        stats_dict = save_stats_dict(x_p, y, y_pred, savedir)
        umap_emb = generate_umap_emb(all_embs, reducer)
        viz_classes(umap_emb, y, savedir)
        viz_pred_classes(umap_emb, y_pred, savedir)
'''

def save_embs_preds_probs(umap_emb, all_embs, y_pred, y_probs, y_probs_full, savedir):
    save_dict = {"umap_emb":umap_emb.tolist(), "all_embs":all_embs, "y_pred":y_pred, "y_probs":y_probs, "y_probs_full":y_probs_full}
    with open(os.path.join(savedir, 'embs_preds_probs.pkl'), 'wb') as handle:
        pickle.dump(save_dict, handle)

############### CC Evaluation ######################
for model_idx in [2]:#range(len(models)):
    print('Model:', models[model_idx])
    reducer = umap.UMAP(random_state=42)
    for dataset_idx in range(len(datasets)):
        Path('{}_{}'.format(datasets[dataset_idx], models[model_idx])).mkdir(exist_ok=True)
        print('Dataset:', datasets[dataset_idx])
        x, y = dl.load_dataset_nx(datasets[dataset_idx])
        x_p, y_p = dl.load_dataset_ptg(datasets[dataset_idx])
        print(len(x_p), len(y_p))
        for aug_idx, augmode in enumerate(["default"]):
            aug_idx_temp = 0
            Path('{}_{}'.format(datasets[dataset_idx], models[model_idx]), augmode).mkdir(exist_ok=True)
            all_embs, y_pred, y_probs, y_probs_full = get_embs_and_preds(model_idx, x_p, aug_idx_temp)
            savedir=Path('{}_{}'.format(datasets[dataset_idx], models[model_idx]), augmode)
            #stats_dict = save_stats_dict(x_p, y, y_pred, savedir)
            umap_emb = generate_umap_emb(all_embs, reducer)
            sample_umap_embeddings(umap_emb, x_p, y_p, y_pred, savedir)
            viz_classes(umap_emb, y, savedir)
            viz_classes_allinone(umap_emb, y, savedir)
            viz_pred_classes(umap_emb, y_pred, savedir)
            viz_pred_classes_allinone(umap_emb, y_pred, savedir)
            save_embs_preds_probs(umap_emb, all_embs, y_pred, y_probs, y_probs_full, savedir)


