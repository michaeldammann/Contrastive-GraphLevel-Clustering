from torch_geometric.data import Dataset
import pickle


class CeiloGraphs(Dataset):
    def __init__(self, root, graphpicklepath, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.all_graphs=[]
        with open(graphpicklepath, 'rb') as handle:
            pickledict = pickle.load(handle)
        for key, value in pickledict.items():
            self.all_graphs.extend(value)


    def len(self):
        return len(self.all_graphs)

    def get(self, idx):
        return self.all_graphs[idx]