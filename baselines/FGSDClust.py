from karateclub import FGSD
from graph_helper.DatasetLoader import DatasetLoader
import networkx as nx
from sklearn.cluster import KMeans

class FGSDClust:

    def __init__(self, dims):
        self.dl = DatasetLoader()
        self.dims = dims

    def fgsdembs(self, dataset):
        all_graphs_nx, _ = self.dl.load_dataset_nx(dataset)
        fgsd = FGSD(hist_bins=self.dims)
        fgsd.fit([nx.convert_node_labels_to_integers(g) for g in all_graphs_nx])
        return fgsd.get_embedding()

    def kmeans_clust(self, k, dataset):
        all_embs = self.fgsdembs(dataset)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(all_embs)

        return all_embs, kmeans.labels_