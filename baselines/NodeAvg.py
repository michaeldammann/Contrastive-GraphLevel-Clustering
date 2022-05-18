import numpy as np
from sklearn.cluster import KMeans
from graph_helper.DatasetLoader import DatasetLoader


class NodeAvg:

    def __init__(self):
        self.dl = DatasetLoader()

    def node_avgs(self, dataset):
        all_graphs_t, _ = self.dl.load_dataset_ptg(dataset)
        all_node_avgs = []
        for graph in all_graphs_t:
            node_feats = graph.x.cpu().detach().numpy()
            all_node_avgs.append(np.mean(node_feats, axis=0))
        return all_node_avgs

    def kmeans_clust(self, k, dataset):
        all_node_avgs = self.node_avgs(dataset)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(all_node_avgs)
        return all_node_avgs, kmeans.labels_
