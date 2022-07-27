from karateclub import GL2Vec, SF
from graph_helper.DatasetLoader import DatasetLoader
import networkx as nx
from sklearn.cluster import KMeans
import numpy as np

class SFGL2VClust:

    def __init__(self, dims):
        self.dl = DatasetLoader()
        self.dims = dims

    def sfgl2vembs(self, dataset):
        all_graphs_nx, _ = self.dl.load_dataset_nx_g2v(dataset)
        sf = SF(dimensions=self.dims/2)
        gl2v = GL2Vec(dimensions=self.dims/2)
        gl2v.fit([nx.convert_node_labels_to_integers(g) for g in all_graphs_nx])
        sf.fit([nx.convert_node_labels_to_integers(g) for g in all_graphs_nx])
        gl2vemb = gl2v.get_embedding()
        sfemb = sf.get_embedding()
        return np.concatenate([gl2vemb, sfemb], axis=1)

    def kmeans_clust(self, k, dataset):
        all_embs = self.sfgl2vembs(dataset)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(all_embs)

        return all_embs, kmeans.labels_