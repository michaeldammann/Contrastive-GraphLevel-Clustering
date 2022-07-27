import networkx as nx
from karateclub import IGE
from graph_helper.DatasetLoader import DatasetLoader
import pickle

dl = DatasetLoader()
datasets = ['constructedgraphs_2']
picklepath='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasetpickles'

for dataset in datasets:
    ds=dl.load_dataset_nx(dataset)
    with open('{}/{}_x.pkl'.format(picklepath, dataset), 'wb') as handle:
        pickle.dump(ds[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('{}/{}_y.pkl'.format(picklepath, dataset), 'wb') as handle:
        pickle.dump(ds[1], handle, protocol=pickle.HIGHEST_PROTOCOL)

