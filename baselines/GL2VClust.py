from karateclub import GL2Vec
from graph_helper.DatasetLoader import DatasetLoader
import networkx as nx
from sklearn.cluster import KMeans

class GL2VClust:

    def __init__(self, dims):
        self.dl = DatasetLoader()
        self.dims = dims

    def gl2vembs(self, dataset):
        all_graphs_nx, _ = self.dl.load_dataset_nx_g2v(dataset)
        gl2v = GL2Vec(dimensions=self.dims)
        gl2v.fit([nx.convert_node_labels_to_integers(g) for g in all_graphs_nx])
        return gl2v.get_embedding()

    def kmeans_clust(self, k, dataset):
        all_embs = self.gl2vembs(dataset)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(all_embs)

        return all_embs, kmeans.labels_