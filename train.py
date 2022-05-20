import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from modules.GraphTransform_All import GraphTransform_All
from modules.GraphTransform_All_TUD import GraphTransform_All_TUD
from modules.GraphTransform_All_molhiv import GraphTransform_All_molhiv
from modules.GraphTransform_NoNodeMask import GraphTransform_NoNodeMask
from modules.gcn import GCN
from utils import yaml_config_hook, save_model
from torch.utils import data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from modules.graph_utils import max_degree_undirected
import torch_geometric
import random
from copy import deepcopy
from torch.utils.data import ConcatDataset
from torch_geometric.datasets import MNISTSuperpixels, GNNBenchmarkDataset,  TUDataset
import time
from datasets.CeiloGraphs import CeiloGraphs
from ogb.graphproppred import PygGraphPropPredDataset
from graph_helper.DatasetLoader import DatasetLoader

def train():
    loss_epoch = 0
    for step in range(len(data_loader_i)):
        x_i = next(data_loader_i)
        x_j = next(data_loader_j)

        #print(x_i.x.float(), x_i.edge_index, x_i.batch)
        optimizer.zero_grad()
        x_i = x_i.to('cuda')
        x_j = x_j.to('cuda')
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"Step [{step}/{len(data_loader_i)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    print('Cuda:')
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.dataset == "CeiloGraphs":
        dataset = CeiloGraphs(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', graphpicklepath=args.graphpicklepath)
        GraphTransformer = GraphTransform_All(drop_nodes_ratio=0.1, subgraph_ratio=0.9, permute_edges_ratio=0.1,
                                              mask_nodes_ratio=0.1, batch_size=args.batch_size)
        class_num = args.ceilo_n_clusters
        num_features = dataset.num_features
    elif args.dataset == 'ogbg-molhiv':
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
        GraphTransformer = GraphTransform_All_molhiv(drop_nodes_ratio=0.1, subgraph_ratio=0.9, permute_edges_ratio=0.1,
                                              mask_nodes_ratio=0.1, batch_size=args.batch_size)
        class_num = 2
        num_features = dataset.num_features
    elif args.dataset == 'TWITTER-Real-Graph-Partial':
        dataset = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name=args.dataset)
        GraphTransformer = GraphTransform_All(drop_nodes_ratio=0.1, subgraph_ratio=0.9, permute_edges_ratio=0.1,
                                                  mask_nodes_ratio=0.1, batch_size=args.batch_size)
        class_num = dataset.num_classes
        num_features = dataset.num_features
    elif args.dataset == "MNISTSuperpixels":
        dataset = MNISTSuperpixels(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/MNISTSuperpixels', train=True)
        GraphTransformer = GraphTransform_All(drop_nodes_ratio=0.1, subgraph_ratio=0.9, permute_edges_ratio=0.1,
                                              mask_nodes_ratio=0.1, batch_size=args.batch_size)
        '''
        dataset_train = MNISTSuperpixels(root='/home/md/PycharmProjects/CC4Graphs/datasets/', train=True)
        dataset_test = MNISTSuperpixels(root='/home/md/PycharmProjects/CC4Graphs/datasets/', train=False)
        dataset = ConcatDataset([dataset_train, dataset_test])
        dataset.num_classes = dataset_train.num_classes
        dataset.num_features = dataset_train.num_features
        '''
        class_num = dataset.num_classes
        num_features = dataset.num_features
    elif args.dataset in ["constructedgraphs_2", "constructedgraphs_4", "constructedgraphs_2nodedeg"]:
        dl = DatasetLoader()
        dataset, y_p = dl.load_dataset_ptg(args.dataset)
        GraphTransformer = GraphTransform_All(drop_nodes_ratio=0.2, subgraph_ratio=0.8, permute_edges_ratio=0.2,
                                              mask_nodes_ratio=0.2, batch_size=args.batch_size)
        class_num = len(np.unique(y_p))
        num_features = len(dataset[0].x[0].cpu().detach().numpy())
        print(class_num, num_features)
    else:
        dataset_pretransform = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name=args.dataset)
        dataset = TUDataset(root='/home/rigel/MDammann/PycharmProjects/CC4Graphs/datasets/', name=args.dataset,
                            transform=torch_geometric.transforms.OneHotDegree(
                                max_degree=max_degree_undirected(dataset_pretransform)))
        GraphTransformer = GraphTransform_All_TUD(drop_nodes_ratio=0.1, subgraph_ratio=0.9, permute_edges_ratio=0.1,
                                              mask_nodes_ratio=0.1, batch_size=args.batch_size)

        class_num = dataset.num_classes
        num_features = dataset.num_features

    '''
    # prepare data
    if args.dataset == "NCI1":
        dataset = TUDataset(root='/home/md/PycharmProjects/CC4Graphs/datasets/', name=args.dataset)
        class_num = dataset.num_classes
        print(class_num)
    else:
        raise NotImplementedError
    '''


    # initialize model
    gnn = GCN(num_features)
    model = network.Network(gnn, args.feature_dim, class_num)
    model = model.to('cuda')
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1
    loss_device = torch.device("cuda")
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)
    # train
    #_,_,data_i, data_j = GraphTransformer.generate_augmentations_i_j(dataset)
    for epoch in range(args.start_epoch, args.epochs):
        '''
        c = list(zip(data_i, data_j))
        random.shuffle(c)
        data_i, data_j = zip(*c)
        '''
        start_time = time.time()
        data_loader_i, data_loader_j, _, _ = GraphTransformer.generate_augmentations_i_j(dataset)
        print("--- %s seconds ---" % (time.time() - start_time))
        #data_loader_i, data_loader_j = DataLoader(data_i, batch_size=args.batch_size), DataLoader(data_j, batch_size=args.batch_size)
        data_loader_i, data_loader_j = iter(data_loader_i), iter(data_loader_j)
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 10 == 0:
            print('Saving')
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader_i)}")
    save_model(args, model, optimizer, args.epochs)
