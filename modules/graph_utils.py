import numpy as np

def max_degree_undirected(dataset):
    max_degree = 0
    for idx, data in enumerate(dataset):
        edge_index = data.edge_index.cpu().detach().numpy()
        unique, counts = np.unique(edge_index[0], return_counts=True)
        data_max_degree = np.max(counts)
        if data_max_degree>max_degree:
            max_degree = data_max_degree

    return max_degree

