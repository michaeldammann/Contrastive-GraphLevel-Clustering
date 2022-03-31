import os
import numpy as np
import torch
import torchvision
import argparse
from modules import transform, resnet, network, contrastive_loss
from modules.GraphTransform_All import GraphTransform_All
from modules.gcn import GCN
from utils import yaml_config_hook, save_model
from torch.utils import data
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import random


def train():
    loss_epoch = 0
    for step in range(len(data_loader_i)):
        x_i = next(data_loader_i)
        x_j = next(data_loader_j)
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

    # prepare data
    if args.dataset == "NCI1":
        dataset = TUDataset(root='/home/md/PycharmProjects/CC4Graphs/datasets/', name="NCI1")
        class_num = dataset.num_classes
    else:
        raise NotImplementedError

    GraphTransformer = GraphTransform_All(drop_nodes_ratio=0.2, subgraph_ratio=0.2, permute_edges_ratio=0.2, mask_nodes_ratio=0.2, batch_size=args.batch_size)

    # initialize model
    gnn = GCN(dataset.num_features)
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
    for epoch in range(args.start_epoch, args.epochs):
        data_loader_i, data_loader_j = GraphTransformer.generate_augmentations_i_j(dataset)
        data_loader_i, data_loader_j = iter(data_loader_i), iter(data_loader_j)
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        if epoch % 10 == 0:
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader_i)}")
    save_model(args, model, optimizer, args.epochs)
