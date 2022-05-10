# Augmentations from https://github.com/Shen-Lab/GraphCL/blob/master/semisupervised_TU/pre-training/tu_dataset.py
import torch
import numpy as np
from copy import deepcopy
import random
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import inspect

class GraphTransform_All:
  def __init__(self, drop_nodes_ratio, subgraph_ratio, permute_edges_ratio, mask_nodes_ratio, batch_size):
    self.drop_nodes_ratio = drop_nodes_ratio
    self.subgraph_ratio = subgraph_ratio
    self.permute_edges_ratio = permute_edges_ratio
    self.mask_nodes_ratio = mask_nodes_ratio
    self.batch_size = batch_size

  def drop_nodes(self, data, aug_ratio):
      node_num, _ = data.x.size()

      _, edge_num = data.edge_index.size()
      drop_num = int(node_num * aug_ratio)

      idx_perm = np.random.permutation(node_num)

      idx_drop = idx_perm[:drop_num]
      idx_nondrop = idx_perm[drop_num:]
      idx_nondrop.sort()
      idx_dict = {idx_nondrop[n]: n for n in list(range(idx_nondrop.shape[0]))}

      edge_index = data.edge_index.numpy()
      adj = torch.zeros((node_num, node_num))
      adj[edge_index[0], edge_index[1]] = 1
      adj = adj[idx_nondrop, :][:, idx_nondrop]
      edge_index = adj.nonzero().t()

      #new_data = Data(x=data.x[idx_nondrop], edge_index=edge_index, y=data.y, num_nodes=data.x[idx_nondrop].size()[0])


      try:
          #data = Data(x=data.x[idx_nondrop], edge_index=edge_index)

          data.edge_index = edge_index
          data.x = data.x[idx_nondrop]
          
      except:
          data = data


      #data.num_nodes = data.x.size()

      return data

  def permute_edges(self, data, aug_ratio):

      node_num, _ = data.x.size()
      _, edge_num = data.edge_index.size()
      permute_num = int(edge_num * aug_ratio)

      edge_index = data.edge_index.numpy()

      idx_add = np.random.choice(node_num, (2, permute_num))

      # idx_add = [[idx_add[0, n], idx_add[1, n]] for n in range(permute_num) if not (idx_add[0, n], idx_add[1, n]) in edge_index]
      # edge_index = [edge_index[n] for n in range(edge_num) if not n in np.random.choice(edge_num, permute_num, replace=False)] + idx_add

      edge_index = np.concatenate(
          (edge_index[:, np.random.choice(edge_num, (edge_num - permute_num), replace=False)], idx_add), axis=1)
      data.edge_index = torch.tensor(edge_index)
      #new_data = Data(x=data.x, edge_index=data.edge_index, y=data.y, num_nodes=data.x.size()[0])
      #data.num_nodes = data.x.size()
      return data

  def subgraph(self, data, aug_ratio):

      node_num, _ = data.x.size()

      _, edge_num = data.edge_index.size()
      sub_num = int(node_num * aug_ratio)

      edge_index = data.edge_index.numpy()

      idx_sub = [np.random.randint(node_num, size=1)[0]]
      idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])

      count = 0
      while len(idx_sub) <= sub_num:
          count = count + 1
          if count > node_num:
              break
          if len(idx_neigh) == 0:
              break
          sample_node = np.random.choice(list(idx_neigh))
          if sample_node in idx_sub:
              continue
          idx_sub.append(sample_node)
          idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))

      idx_drop = [n for n in range(node_num) if not n in idx_sub]
      idx_nondrop = idx_sub
      data.x = data.x[idx_nondrop]
      idx_dict = {idx_nondrop[n]: n for n in list(range(len(idx_nondrop)))}

      edge_index = data.edge_index.numpy()
      adj = torch.zeros((node_num, node_num))
      adj[edge_index[0], edge_index[1]] = 1
      # adj[list(range(node_num)), list(range(node_num))] = 1
      adj = adj[idx_nondrop, :][:, idx_nondrop]
      edge_index = adj.nonzero().t()

      # edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)] + [[n, n] for n in idx_nondrop]
      data.edge_index = edge_index
      #new_data = Data(x=deepcopy(data.x), edge_index=deepcopy(edge_index), y=data.y, num_nodes=data.x.size()[0])
      #data.num_nodes = data.x.size()

      return data

  def mask_nodes(self, data, aug_ratio):
      node_num, feat_dim = data.x.size()
      mask_num = int(node_num * aug_ratio)

      data.x=data.x.type('torch.FloatTensor')
      #temp_tensor = data.x.type('torch.FloatTensor')
      #token = temp_tensor.mean(dim=0)
      token = data.x.mean(dim=0)

      idx_mask = np.random.choice(node_num, mask_num, replace=False)
      data.x[idx_mask] = torch.tensor(token, dtype=torch.float32)
      #new_data = Data(x=data.x, edge_index=data.edge_index, y=data.y, num_nodes=data.x.size()[0])
      #data.num_nodes = data.x.size()
      return data

  def generate_augmentations_i_j(self, dataset):
      all_data = [elem for elem in dataset]
      random.shuffle(all_data)

      #Determine cutoff index to avoid incomplete batches
      cutoff_at = len(all_data)-(len(all_data)%self.batch_size)
      all_data = all_data[:cutoff_at]

      all_data_i, all_data_j = deepcopy(all_data), deepcopy(all_data)

      for idx, elem in enumerate(all_data):
          ri, rj = np.random.randint(5), np.random.randint(5)
          if ri == 0:
              all_data_i[idx]=self.drop_nodes(all_data_i[idx], self.drop_nodes_ratio)
          elif ri == 1:
              all_data_i[idx]=self.subgraph(all_data_i[idx], self.subgraph_ratio)
          elif ri == 2:
              all_data_i[idx]=self.permute_edges(all_data_i[idx], self.permute_edges_ratio)
          elif ri == 3:
              all_data_i[idx]=self.mask_nodes(all_data_i[idx], self.mask_nodes_ratio)
          # elif ri == 4: identity

          if rj == 0:
              all_data_j[idx]=self.drop_nodes(all_data_j[idx], self.drop_nodes_ratio)
          elif rj == 1:
              all_data_j[idx]=self.subgraph(all_data_j[idx], self.subgraph_ratio)
          elif rj == 2:
              all_data_j[idx]=self.permute_edges(all_data_j[idx], self.permute_edges_ratio)
          elif rj == 3:
              all_data_j[idx]=self.mask_nodes(all_data_j[idx], self.mask_nodes_ratio)
          # elif rj == 4: identity

      return (DataLoader(all_data_i, batch_size=self.batch_size), DataLoader(all_data_j, batch_size=self.batch_size), all_data_i, all_data_j)

  def apply_aug(self, data, randint):
      if randint == 0:
          return self.drop_nodes(data, self.drop_nodes_ratio)
      elif randint == 1:
          return self.subgraph(data, self.subgraph_ratio)
      elif randint == 2:
          return self.permute_edges(data, self.permute_edges_ratio)
      elif randint == 3:
          return self.mask_nodes(data, self.mask_nodes_ratio)
      elif randint == 4:
          return data

  def generate_augmentations_i_j_fast(self, dataset):


      all_data = [elem for elem in dataset]
      random.shuffle(all_data)

      #Determine cutoff index to avoid incomplete batches
      cutoff_at = len(all_data)-(len(all_data)%self.batch_size)
      all_data = all_data[:cutoff_at]

      all_data_i, all_data_j = deepcopy(all_data), deepcopy(all_data)

      ri_list = [np.random.randint(5) for i in range(len(all_data))]
      rj_list = [np.random.randint(5) for j in range(len(all_data))]


      all_data_i = [self.apply_aug(data_aug_tuple[0], data_aug_tuple[1]) for data_aug_tuple in zip(all_data_i, ri_list)]
      all_data_j = [self.apply_aug(data_aug_tuple[0], data_aug_tuple[1]) for data_aug_tuple in zip(all_data_j, rj_list)]


      return (DataLoader(all_data_i, batch_size=self.batch_size), DataLoader(all_data_j, batch_size=self.batch_size), all_data_i, all_data_j)

