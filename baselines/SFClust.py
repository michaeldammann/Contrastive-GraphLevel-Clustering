from karateclub import SF
from graph_helper.DatasetLoader import DatasetLoader
import networkx as nx
from sklearn.cluster import KMeans

class SFClust:

    def __init__(self, dims):
        self.dl = DatasetLoader()
        self.dims = dims

    def sfembs(self, dataset):
        all_graphs_nx, _ = self.dl.load_dataset_nx(dataset)
        sf = SF(dimensions=self.dims)
        sf.fit([nx.convert_node_labels_to_integers(g) for g in all_graphs_nx])
        return sf.get_embedding()

    def kmeans_clust(self, k, dataset):
        all_embs = self.sfembs(dataset)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(all_embs)

        return all_embs, kmeans.labels_