import pickle
from os import path
import cv2
import numpy as np
from matplotlib.colors import to_rgb
from pathlib import Path
import json
import matplotlib.pyplot as plt
from graph_helper.DatasetLoader import DatasetLoader
from networkx import diameter, average_clustering
import math

ceilographs_dir = '/home/rigel/MDammann/PycharmProjects/CC4Graphs/ceilo_graph_generation/graphs_processed.pickle'
img_dir = '/media/rigel/Data/DWD/allimgs'
savepath_base = '/home/rigel/MDammann/PycharmProjects/CC4Graphs/baselines'
'''
datasets = ['ceilographs8_location']
'''
datasets = ['ceilographs4_year', 'ceilographs4_month', 'ceilographs4_location', 'ceilographs4_season', 'ceilographs4_nodenumber', 'ceilographs4_edgenumber', 'ceilographs4_features',
            'ceilographs8_year', 'ceilographs8_month', 'ceilographs8_location', 'ceilographs8_season', 'ceilographs8_nodenumber', 'ceilographs8_edgenumber', 'ceilographs8_features',
            'ceilographs16_year', 'ceilographs16_month', 'ceilographs16_location', 'ceilographs16_season', 'ceilographs16_nodenumber', 'ceilographs16_edgenumber', 'ceilographs16_features']



with open(ceilographs_dir, 'rb') as handle:
    ceilographs = pickle.load(handle)

#print(ceilographs)
clustercolors = {0:'blue', 1:'red', 2:'green', 3:'orange', 4:'cyan', 5:'magenta', 6:'yellow', 7:'darkred', 8:'teal',
                 9:'darkviolet', 10:'chocolate', 11:'pink'}

all_keys = []#list(ceilographs.keys())
for k, v in ceilographs.items():
    for v_n in v:
        all_keys.append(k)

#print(ceilographs['20140725_hohenpeissenberg_CHM060028_000.nc_original._'])
#keys_idx = 460
#bild laden

subgraph_indices = []
for k, v in ceilographs.items():
    for idx, v_n in enumerate(v):
        subgraph_indices.append(idx)


def visualize_graph(graph_index, savedir):
    im = cv2.imread(path.join(img_dir, all_keys[graph_index][:-1] + 'png'))
    im_dims = (im.shape[1], im.shape[0])  # im.size #(w,h)

    # quadrate in originalbild zeichnen
    all_nodes = ceilographs[all_keys[graph_index]][subgraph_indices[graph_index]].x.cpu().detach().numpy()

    all_edges = ceilographs[all_keys[graph_index]][subgraph_indices[graph_index]].edge_index.cpu().detach().numpy()
    all_edges_transf = [(all_edges[0][idx], all_edges[1][idx]) for idx in range(len(all_edges[0]))]
    all_edges_final = []
    for edge in all_edges_transf:
        if not (edge[1], edge[0]) in all_edges_final:
            all_edges_final.append(edge)
    all_node_centercoords = []

    for node in all_nodes:
        x_min_rel, x_max_rel, y_min_rel, y_max_rel = node[12:16]
        cluster = np.argmax(node[0:12])
        x_min, x_max, y_min, y_max = int(x_min_rel * im_dims[0]), int(x_max_rel * im_dims[0]), int(
            y_min_rel * im_dims[1]), int(y_max_rel * im_dims[1])
        center_coords = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        all_node_centercoords.append(center_coords)
        # draw.rounded_rectangle([x_min, y_min, x_max, y_max], width=2, outline=clustercolors[cluster])
        rgb_color = to_rgb(clustercolors[cluster])
        color_tuple = (int(rgb_color[2] * 255), int(rgb_color[1] * 255), int(rgb_color[0] * 255))
        cv2.rectangle(im, (x_min, y_min), (x_max, y_max), color_tuple, 1, lineType=cv2.LINE_AA)
        cv2.line(im, (x_min, y_min), (x_max, y_max), color_tuple, thickness=1, lineType=cv2.LINE_AA)
        '''
        font = ImageFont.truetype("sans-serif.ttf", 16)
        # draw.text((x, y),"Sample Text",(r,g,b))
        draw.text(((x_max+x_min)/2, (y_max+y_min)/2), "Sample Text", (255, 255, 255), font=font)
        '''

    # knoten verbinden

    for edge in all_edges_final:
        x0 = all_node_centercoords[edge[0]][0]
        y0 = all_node_centercoords[edge[0]][1]
        x1 = all_node_centercoords[edge[1]][0]
        y1 = all_node_centercoords[edge[1]][1]
        cv2.line(im, (x0, y0), (x1, y1), (128, 128, 128), thickness=2, lineType=cv2.LINE_AA)
        # draw.line([(x0, y0), (x1, y1)], width=3, fill='grey')

    for node in all_nodes:
        x_min_rel, x_max_rel, y_min_rel, y_max_rel = node[12:16]
        cluster = np.argmax(node[0:12])
        x_min, x_max, y_min, y_max = int(x_min_rel * im_dims[0]), int(x_max_rel * im_dims[0]), int(
            y_min_rel * im_dims[1]), int(y_max_rel * im_dims[1])
        center_coords = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        rgb_color = to_rgb(clustercolors[cluster])
        color_tuple = (int(rgb_color[2] * 255), int(rgb_color[1] * 255), int(rgb_color[0] * 255))
        circle_radius = 7
        cv2.circle(im, center_coords, radius=circle_radius, color=color_tuple, lineType=cv2.LINE_AA, thickness=-1)
        cv2.circle(im, center_coords, radius=circle_radius, color=(0, 0, 0),
                   lineType=cv2.LINE_AA, thickness=1)

    cv2.imwrite(savedir, im)

def viz_top_n(dataset_idx, n):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    umap_emb = embs_preds_probs["umap_emb"]
    all_embs = embs_preds_probs["all_embs"]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    y_probs_full = embs_preds_probs["y_probs_full"]
    Path(savepath_base, datasets[dataset_idx] + '_CC', 'default', 'top_examples').mkdir(parents=True, exist_ok=True)
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    for cluster_n in range(n_clusters):
        Path(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'top_examples', str(cluster_n)).mkdir(parents=True, exist_ok=True)
        y_probs_masked = [y_probs[i] if y_pred[i]==cluster_n else 0 for i in range(len(y_probs))]
        sorted_indices = np.flip(np.argsort(y_probs_masked))
        cluster_dict={}
        for top_n in range(n):
            visualize_graph(sorted_indices[top_n], path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'top_examples', str(cluster_n), str(top_n)+'.png'))
            cluster_dict[str(top_n)]={"name":all_keys[sorted_indices[top_n]],"umap_emb":[umap_emb[sorted_indices[top_n]][0], umap_emb[sorted_indices[top_n]][1]], "y_prob":float(y_probs[sorted_indices[top_n]]), "y_prob_full":y_probs_full[sorted_indices[top_n]].tolist()}
        with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'top_examples', str(cluster_n), 'clusterdict.json'), 'w') as f:
            json.dump(cluster_dict, f)

def viz_flop_n(dataset_idx, n):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    umap_emb = embs_preds_probs["umap_emb"]
    y_probs_full = embs_preds_probs["y_probs_full"]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    Path(savepath_base, datasets[dataset_idx] + '_CC', 'default', 'flop_examples').mkdir(parents=True, exist_ok=True)
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    for cluster_n in range(n_clusters):
        Path(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'flop_examples', str(cluster_n)).mkdir(parents=True, exist_ok=True)
        y_probs_masked = [y_probs[i] if y_pred[i]==cluster_n else 1.1 for i in range(len(y_probs))]
        sorted_indices = np.argsort(y_probs_masked)
        cluster_dict={}
        for top_n in range(n):
            visualize_graph(sorted_indices[top_n], path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'flop_examples', str(cluster_n), str(top_n)+'.png'))
            cluster_dict[str(top_n)]={"name":all_keys[sorted_indices[top_n]],
                                      "umap_emb":[umap_emb[sorted_indices[top_n]][0], umap_emb[sorted_indices[top_n]][1]],
                                      "y_prob":float(y_probs[sorted_indices[top_n]]),
                                      "y_prob_full":y_probs_full[sorted_indices[top_n]].tolist()}
        with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'flop_examples', str(cluster_n), 'clusterdict.json'), 'w') as f:
            json.dump(cluster_dict, f)



#visualize_graph(30, "test.png")
def violin_plot(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    for cluster_n in range(n_clusters):
        y_probs_cluster_n = []
        for idx, elem in enumerate(y_probs):
            if y_pred[idx]==cluster_n:
                y_probs_cluster_n.append(y_probs[idx])
        all_y_probs.append(y_probs_cluster_n)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 21})
    ax.set_title(str(n_clusters)+' clusters')
    ax.set_ylabel('Cluster probability')
    ax.set_xlabel('Cluster')
    ax.violinplot(all_y_probs)
    plt.yticks([0.1*i for i in range(11)])
    plt.xticks([i for i in range(1,n_clusters+1)])
    plt.savefig(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', str(n_clusters)+'_clusters.png'))

def cluster_distrib(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    for cluster_n in range(n_clusters):
        y_probs_cluster_n = []
        for idx, elem in enumerate(y_probs):
            if y_pred[idx]==cluster_n:
                y_probs_cluster_n.append(y_probs[idx])
        all_y_probs.append(y_probs_cluster_n)
    x = [i for i in range(1,n_clusters+1)]
    y = [len(elems) for elems in all_y_probs]
    plt.rcParams.update({'font.size': 21})
    fig = plt.figure(figsize=(10, 7))

    # creating the bar plot
    plt.bar(x, y, width=0.5)

    plt.xlabel("Cluster")
    plt.ylabel("Number of samples")
    plt.xticks([i for i in range(1,n_clusters+1)])
    plt.yticks([250*i for i in range(0,14)])
    plt.title("Cluster distribution")
    plt.savefig(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', str(n_clusters)+'_clusterdistrib.png'))

def cluster_coeff(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    dl = DatasetLoader()
    ds_nx = dl.load_dataset_nx(datasets[dataset_idx])[0]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    graphs_per_cluster = [[] for i in range(n_clusters)]
    for graph_idx in range(len(ds_nx)):
        graphs_per_cluster[y_pred[graph_idx]].append(ds_nx[graph_idx])
    coeffs_per_cluster = [[] for i in range(n_clusters)]
    for clust in range(len(graphs_per_cluster)):
        for single_graph in graphs_per_cluster[clust]:
            coeffs_per_cluster[clust].append(average_clustering(single_graph))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    ax.set_title(str(n_clusters) + ' clusters')
    ax.set_ylabel('Avg. clust. coeff. distrib.')
    ax.set_xlabel('Cluster')
    ax.violinplot(coeffs_per_cluster, showmeans=True)
    plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(1, n_clusters + 1)])
    plt.savefig(path.join(savepath_base, datasets[dataset_idx] + '_CC', 'default', str(n_clusters) + '_clustercoeff.png'))

def cluster_diameter(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    dl = DatasetLoader()
    ds_nx = dl.load_dataset_nx(datasets[dataset_idx])[0]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    graphs_per_cluster = [[] for i in range(n_clusters)]
    for graph_idx in range(len(ds_nx)):
        graphs_per_cluster[y_pred[graph_idx]].append(ds_nx[graph_idx])
    coeffs_per_cluster = [[] for i in range(n_clusters)]
    for clust in range(len(graphs_per_cluster)):
        for single_graph in graphs_per_cluster[clust]:
            coeffs_per_cluster[clust].append(diameter(single_graph))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    ax.set_title(str(n_clusters) + ' clusters')
    ax.set_ylabel('Diameter distrib.')
    ax.set_xlabel('Cluster')
    ax.violinplot(coeffs_per_cluster, showmeans=True)
    #plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(1, n_clusters + 1)])
    plt.savefig(path.join(savepath_base, datasets[dataset_idx] + '_CC', 'default', str(n_clusters) + '_clusterdiameter.png'))

def cluster_nodes(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    dl = DatasetLoader()
    ds_pt = dl.load_dataset_ptg(datasets[dataset_idx])[0]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    graphs_per_cluster = [[] for i in range(n_clusters)]
    for graph_idx in range(len(ds_pt)):
        graphs_per_cluster[y_pred[graph_idx]].append(ds_pt[graph_idx])
    nodenumber_per_cluster = [[] for i in range(n_clusters)]
    for clust in range(len(graphs_per_cluster)):
        for single_graph in graphs_per_cluster[clust]:
            nodenumber_per_cluster[clust].append(len(single_graph.x.cpu().detach().numpy().tolist()))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    ax.set_title(str(n_clusters) + ' clusters')
    ax.set_ylabel('Number of nodes distrib.')
    ax.set_xlabel('Cluster')
    ax.set_ylim([0,35])
    ax.violinplot(nodenumber_per_cluster, showmeans=True)
    #plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(1, n_clusters + 1)])
    plt.savefig(path.join(savepath_base, datasets[dataset_idx] + '_CC', 'default', str(n_clusters) + '_clusternodes.png'))

def cluster_edges(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    dl = DatasetLoader()
    ds_pt = dl.load_dataset_ptg(datasets[dataset_idx])[0]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    graphs_per_cluster = [[] for i in range(n_clusters)]
    for graph_idx in range(len(ds_pt)):
        graphs_per_cluster[y_pred[graph_idx]].append(ds_pt[graph_idx])
    nodenumber_per_cluster = [[] for i in range(n_clusters)]
    for clust in range(len(graphs_per_cluster)):
        for single_graph in graphs_per_cluster[clust]:
            nodenumber_per_cluster[clust].append(len(single_graph.edge_index.cpu().detach().numpy().tolist()[0])/2)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    ax.set_title(str(n_clusters) + ' clusters')
    ax.set_ylabel('Number of edges distrib.')
    ax.set_xlabel('Cluster')
    ax.violinplot(nodenumber_per_cluster, showmeans=True)
    #plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(1, n_clusters + 1)])
    plt.savefig(path.join(savepath_base, datasets[dataset_idx] + '_CC', 'default', str(n_clusters) + '_clusteredges.png'))

def cluster_segmentfeatures(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    dl = DatasetLoader()
    ds_pt = dl.load_dataset_ptg(datasets[dataset_idx])[0]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    graphs_per_cluster = [[] for i in range(n_clusters)]
    for graph_idx in range(len(ds_pt)):
        graphs_per_cluster[y_pred[graph_idx]].append(ds_pt[graph_idx])
    segmentfeature_per_cluster = [[] for i in range(n_clusters)]
    for clust in range(len(graphs_per_cluster)):
        for single_graph in graphs_per_cluster[clust]:
            segmentfeature_per_cluster[clust].append(np.argmax(np.mean(single_graph.x.cpu().detach().numpy()[:,:12], axis=0))+1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    ax.set_title(str(n_clusters) + ' clusters')
    ax.set_ylabel('Segment cluster distrib.')
    ax.set_xlabel('Cluster')
    ax.violinplot(segmentfeature_per_cluster, showextrema=False)
    #plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(1, n_clusters + 1)])
    plt.yticks([i for i in range(1, 13)])
    plt.savefig(path.join(savepath_base, datasets[dataset_idx] + '_CC', 'default', str(n_clusters) + '_clustersegments.png'))

def cluster_segmentfeatures_barplot(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    dl = DatasetLoader()
    ds_pt = dl.load_dataset_ptg(datasets[dataset_idx])[0]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    graphs_per_cluster = [[] for i in range(n_clusters)]
    for graph_idx in range(len(ds_pt)):
        graphs_per_cluster[y_pred[graph_idx]].append(ds_pt[graph_idx])
    segmentfeature_per_cluster = [[0 for j in range(12)] for i in range(n_clusters)]
    for clust in range(len(graphs_per_cluster)):
        for single_graph in graphs_per_cluster[clust]:
            segmentfeature_per_cluster[clust][np.argmax(np.mean(single_graph.x.cpu().detach().numpy()[:,:12], axis=0))]+=1
    #y_limit = int(math.ceil(max([len(clustary) for clustary in segmentfeature_per_cluster[clust]])/100.))*100
    y_limit = int(math.ceil(max([max(elem) for elem in segmentfeature_per_cluster]) / 100.)) * 100
    for clust in range(len(graphs_per_cluster)):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
        plt.rcParams.update({'font.size': 20})
        ax.set_title(str(n_clusters) + ' clusters')
        ax.set_ylabel('Number of graphs')
        ax.set_xlabel('Dominant segment cluster')
        ax.set_ylim([0, y_limit])
        barlist=ax.bar([i for i in range(1,13)], segmentfeature_per_cluster[clust])
        for idx, bar in enumerate(barlist):
            barlist[idx].set_color(clustercolors[idx])
        #plt.yticks([0.1 * i for i in range(11)])
        #plt.xticks([i for i in range(1, n_clusters + 1)])
        plt.xticks([i for i in range(1, 13)])
        plt.savefig(path.join(savepath_base, datasets[dataset_idx] + '_CC', 'default', str(n_clusters) + '_clustersegmentsbar_{}.png'.format(str(clust))))

def cluster_time(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    dl = DatasetLoader()
    ds_pt = dl.load_dataset_ptg(datasets[dataset_idx])[0]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    graphs_per_cluster = [[] for i in range(n_clusters)]
    for graph_idx in range(len(ds_pt)):
        graphs_per_cluster[y_pred[graph_idx]].append(ds_pt[graph_idx])
    segmentfeature_per_cluster = [[] for i in range(n_clusters)]
    for clust in range(len(graphs_per_cluster)):
        for single_graph in graphs_per_cluster[clust]:
            if np.mean(single_graph.x.cpu().detach().numpy()[:,12:14]) <= 1.0:
                segmentfeature_per_cluster[clust].append(np.mean(single_graph.x.cpu().detach().numpy()[:,12:14]))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    ax.set_title(str(n_clusters) + ' clusters')
    ax.set_ylabel('Relative daytime distrib.')
    ax.set_xlabel('Cluster')
    ax.set_ylim([0., 1.])
    ax.violinplot(segmentfeature_per_cluster, showmeans=True)
    #plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(1, n_clusters + 1)])
    plt.savefig(path.join(savepath_base, datasets[dataset_idx] + '_CC', 'default', str(n_clusters) + '_clustertime.png'))

def cluster_height(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    dl = DatasetLoader()
    ds_pt = dl.load_dataset_ptg(datasets[dataset_idx])[0]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    graphs_per_cluster = [[] for i in range(n_clusters)]
    for graph_idx in range(len(ds_pt)):
        graphs_per_cluster[y_pred[graph_idx]].append(ds_pt[graph_idx])
    segmentfeature_per_cluster = [[] for i in range(n_clusters)]
    for clust in range(len(graphs_per_cluster)):
        for single_graph in graphs_per_cluster[clust]:
            segmentfeature_per_cluster[clust].append(1.-np.mean(single_graph.x.cpu().detach().numpy()[:,14:]))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    ax.set_title(str(n_clusters) + ' clusters')
    ax.set_ylabel('Relative altitude distrib.')
    ax.set_xlabel('Cluster')
    ax.set_ylim([0., 1.])
    ax.violinplot(segmentfeature_per_cluster, showmeans=True)
    #plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(1, n_clusters + 1)])
    plt.savefig(path.join(savepath_base, datasets[dataset_idx] + '_CC', 'default', str(n_clusters) + '_clusterheight.png'))

def cluster_time_all(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    dl = DatasetLoader()
    ds_pt = dl.load_dataset_ptg(datasets[dataset_idx])[0]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    graphs_per_cluster = [[] for i in range(n_clusters)]
    for graph_idx in range(len(ds_pt)):
        graphs_per_cluster[y_pred[graph_idx]].append(ds_pt[graph_idx])
    segmentfeature_per_cluster = [[] for i in range(n_clusters)]
    for clust in range(len(graphs_per_cluster)):
        for single_graph in graphs_per_cluster[clust]:
            all_nodes = single_graph.x.cpu().detach().numpy()[:, 12:14]
            for node in all_nodes:
                segmentfeature_per_cluster[clust].append(np.mean(node))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    ax.set_title(str(n_clusters) + ' clusters')
    ax.set_ylabel('Relative time distrib.')
    ax.set_xlabel('Cluster')
    ax.violinplot(segmentfeature_per_cluster, showmeans=True)
    #plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(1, n_clusters + 1)])
    plt.savefig(path.join(savepath_base, datasets[dataset_idx] + '_CC', 'default', str(n_clusters) + '_clustertimeall.png'))

def cluster_height_all(dataset_idx):
    with open(path.join(savepath_base, datasets[dataset_idx]+'_CC', 'default', 'embs_preds_probs.pkl'), 'rb') as handle:
        embs_preds_probs = pickle.load(handle)
    dl = DatasetLoader()
    ds_pt = dl.load_dataset_ptg(datasets[dataset_idx])[0]
    y_pred = embs_preds_probs["y_pred"]
    y_probs = embs_preds_probs["y_probs"]
    all_y_probs = []
    n_clusters = int(datasets[dataset_idx].split('_')[0][11:])
    graphs_per_cluster = [[] for i in range(n_clusters)]
    for graph_idx in range(len(ds_pt)):
        graphs_per_cluster[y_pred[graph_idx]].append(ds_pt[graph_idx])
    segmentfeature_per_cluster = [[] for i in range(n_clusters)]
    for clust in range(len(graphs_per_cluster)):
        for single_graph in graphs_per_cluster[clust]:
            all_nodes = single_graph.x.cpu().detach().numpy()[:, 14:]
            for node in all_nodes:
                segmentfeature_per_cluster[clust].append(1.-np.mean(node))
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    ax.set_title(str(n_clusters) + ' clusters')
    ax.set_ylabel('Relative height distrib.')
    ax.set_xlabel('Cluster')
    ax.violinplot(segmentfeature_per_cluster, showmeans=True)
    #plt.yticks([0.1 * i for i in range(11)])
    plt.xticks([i for i in range(1, n_clusters + 1)])
    plt.savefig(path.join(savepath_base, datasets[dataset_idx] + '_CC', 'default', str(n_clusters) + '_clusterheightall.png'))

n_viz=25
for d_idx in range(len(datasets)):
    #viz_top_n(d_idx, n_viz)
    #viz_flop_n(d_idx, n_viz)
    #violin_plot(d_idx)
    cluster_distrib(d_idx)
    '''
    cluster_coeff(d_idx)
    cluster_diameter(d_idx)
    cluster_coeff(d_idx)
    cluster_edges(d_idx)
    cluster_nodes(d_idx)
    cluster_height_all(d_idx)
    cluster_time_all(d_idx)
    cluster_height_all(d_idx)

    cluster_segmentfeatures(d_idx)
    cluster_time(d_idx)
    cluster_height(d_idx)
    cluster_segmentfeatures(d_idx)
    
    cluster_segmentfeatures_barplot(d_idx)
    cluster_segmentfeatures_barplot(d_idx)

    cluster_time(d_idx)
    cluster_height(d_idx)
    cluster_time(d_idx)
    '''




