import os
import numpy as np
import torch
import argparse
from modules import network, contrastive_loss
from modules.GraphTransform_All import GraphTransform_All
from modules.gcn import GCN
from utils import yaml_config_hook
import random
import pickle

from datasetloader.DatasetLoader import DatasetLoader

def inference():
    loss_epoch = 0
    all_cluster, all_reps = [], []
    for step in range(len(data_loader_i)):
        x_i = next(data_loader_i)
        x_j = next(data_loader_j)

        optimizer.zero_grad()
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)
        z, _, c, _ = model(x_i, x_j)
        z = z.cpu().detach().numpy().tolist()
        c = c.cpu().detach().numpy().tolist()
        all_cluster.extend(c)
        all_reps.extend(z)
    return all_cluster, all_reps


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
    # initialize model
    gnn = GCN(num_features)
    model = network.Network(gnn, args.feature_dim, class_num)
    model = model.to(args.device)
    # optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.inference_epoch))
    checkpoint = torch.load(model_fp)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    args.start_epoch = checkpoint['epoch'] + 1

    loss_device = torch.device(args.device)
    criterion_instance = contrastive_loss.InstanceLoss(args.batch_size, args.instance_temperature, loss_device).to(
        loss_device)
    criterion_cluster = contrastive_loss.ClusterLoss(class_num, args.cluster_temperature, loss_device).to(loss_device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    data_loader_i, data_loader_j, _, _ = GraphTransformer.generate_augmentations_i_j_noaug(dataset)
    data_loader_i, data_loader_j = iter(data_loader_i), iter(data_loader_j)
    clusters, reps = inference()
    os.makedirs(args.inference_save_path, exist_ok=True)
    with open(os.path.join(args.inference_save_path, 'clusters.pkl'), 'wb') as handle:
        pickle.dump(clusters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.inference_save_path, 'reps.pkl'), 'wb') as handle:
        pickle.dump(reps, handle, protocol=pickle.HIGHEST_PROTOCOL)

