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
        all_graphs_nx_g2v, _ = self.dl.load_dataset_nx_g2v(dataset)
        all_graphs_nx_sf, _ = self.dl.load_dataset_nx(dataset)
        sf = SF(dimensions=int(self.dims/2))
        gl2v = GL2Vec(dimensions=int(self.dims/2))
        gl2v.fit([nx.convert_node_labels_to_integers(g) for g in all_graphs_nx_g2v])
        sf.fit([nx.convert_node_labels_to_integers(g) for g in all_graphs_nx_sf])
        gl2vemb = gl2v.get_embedding()
        sfemb = sf.get_embedding()
        wholeemb = np.concatenate([gl2vemb, sfemb], axis=1)
        print(wholeemb.shape)
        print(gl2vemb[0])
        print(sfemb[0])
        print(wholeemb[0])
        return wholeemb

    def kmeans_clust(self, k, dataset):
        all_embs = self.sfgl2vembs(dataset)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(all_embs)

        return all_embs, kmeans.labels_