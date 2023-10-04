import os
import numpy as np
import torch
import argparse
from modules import network, contrastive_loss
from modules.GraphTransform_All import GraphTransform_All
from modules.gcn import GCN
from utils import yaml_config_hook, save_model
import random

from datasetloader.DatasetLoader import DatasetLoader
from pathlib import Path


def train():
    loss_epoch = 0
    for step in range(len(data_loader_i)):
        x_i = next(data_loader_i)
        x_j = next(data_loader_j)

        optimizer.zero_grad()
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        if args.used_loss == "instanceonly":
            loss = loss_instance
        elif args.used_loss == "clusteronly":
            loss = loss_cluster
        else:
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

    if args.dataset in ["ogbg-molhiv", "ogbg-ppa", "TWITTER-Real-Graph-Partial", "Yeast", "PROTEINS", "DD",
                        "reddit_threads", "twitch_egos", "REDDIT-BINARY", "TRIANGLES", "MNISTSuperpixels"]:
        dl = DatasetLoader(args.dataset_dir)
        dataset, y_p = dl.load_dataset_ptg(args.dataset)
        GraphTransformer = GraphTransform_All(drop_nodes_ratio=args.drop_nodes_ratio,
                                              subgraph_ratio=args.subgraph_ratio,
                                              permute_edges_ratio=args.permute_edges_ratio,
                                              mask_nodes_ratio=args.mask_nodes_ratio, batch_size=args.batch_size)
        class_num = len(np.unique(y_p)) if args.n_clusters < 0 else args.n_clusters
        num_features = len(dataset[0].x[0].cpu().detach().numpy())
    else:
        raise NotImplementedError
    epoch_stats = Path(args.model_path, 'epoch_stats.txt')
    epoch_stats.touch(exist_ok=True)
    # initialize model
    gnn = GCN(num_features)
    model = network.Network(gnn, args.feature_dim, class_num)
    model = model.to(args.device)
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.reload:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    loss_device = torch.device(args.device)
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    for epoch in range(args.start_epoch, args.epochs):
        if args.augmode == "subgraphonly":
            data_loader_i, data_loader_j, _, _ = GraphTransformer.generate_augmentations_i_j_subgraphonly(dataset)
        elif args.augmode == "masknodesonly":
            data_loader_i, data_loader_j, _, _ = GraphTransformer.generate_augmentations_i_j_masknodesonly(dataset)
        elif args.augmode == "dropnodesonly":
            data_loader_i, data_loader_j, _, _ = GraphTransformer.generate_augmentations_i_j_dropnodesonly(dataset)
        elif args.augmode == "permuteedgesonly":
            data_loader_i, data_loader_j, _, _ = GraphTransformer.generate_augmentations_i_j_permuteedgesonly(dataset)
        elif args.augmode == "noaug":
            data_loader_i, data_loader_j, _, _ = GraphTransformer.generate_augmentations_i_j_noaug(dataset)
        else:
            data_loader_i, data_loader_j, _, _ = GraphTransformer.generate_augmentations_i_j(dataset)
        data_loader_i, data_loader_j = iter(data_loader_i), iter(data_loader_j)
        lr = optimizer.param_groups[0]["lr"]
        loss_epoch = train()
        with open(Path(args.model_path, 'epoch_stats.txt'), "a+") as f:
            f.write(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader_i)}\n")
        if epoch % args.checkpoint_interval == 0:
            print('Saving')
            save_model(args, model, optimizer, epoch)
        print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader_i)}")
    save_model(args, model, optimizer, args.epochs)
