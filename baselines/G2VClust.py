from karateclub import Graph2Vec
from graph_helper.DatasetLoader import DatasetLoader
import networkx as nx
from sklearn.cluster import KMeans

class G2VClust:

    def __init__(self, dims):
        self.dl = DatasetLoader()
        self.dims = dims

    def g2vembs(self, dataset):
        all_graphs_nx, _ = self.dl.load_dataset_nx(dataset)
        g2v = Graph2Vec(dimensions=self.dims)
        g2v.fit([nx.convert_node_labels_to_integers(g) for g in all_graphs_nx])
        return g2v.get_embedding()

    def kmeans_clust(self, k, dataset):
        all_embs = self.g2vembs(dataset)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(all_embs)

        return all_embs, kmeans.labels_