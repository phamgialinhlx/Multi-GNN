import torch
import pandas as pd
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero
from training import get_model
from torch_geometric.nn import to_hetero, summary
from data_util import GraphData, HeteroData, z_norm, create_hetero_obj
from util import create_parser, set_seed, logger_setup
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score
import numpy as np
import wandb
import logging
import os
import sys
import tqdm
import json
import time

script_start = time.time()

def create_loader(path, args):
    transaction_file = path
    df_edges = pd.read_csv(transaction_file)
    
    print(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()
    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())
    
    print(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    print(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    print(f"Number of transactions = {df_edges.shape[0]}")

    
    edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    node_features = ['Feature']
    
    print(f'Edge features being used: {edge_features}')
    print(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    data = GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, timestamps=timestamps)
    
    if args.ports:
        logging.info(f"Start: adding ports")
        data.add_ports()
        logging.info(f"Done: adding ports")
    if args.tds:
        logging.info(f"Start: adding time-deltas")
        data.add_time_deltas()
        logging.info(f"Done: adding time-deltas")

    #Normalize data
    data.x = z_norm(data.x)
    if not args.model == 'rgcn':
        data.edge_attr = z_norm(data.edge_attr)
    else:
        data.edge_attr[:, :-1] = z_norm(data.edge_attr[:, :-1])
    if args.reverse_mp:
        data = create_hetero_obj(data.x,  data.y,  data.edge_index,  data.edge_attr, data.timestamps, args)

    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    add_arange_ids([data])

    return LinkNeighborLoader(data, num_neighbors=args.num_neighs, edge_label_index=data.edge_index,
                                edge_label=data.y, batch_size=args.batch_size, shuffle=False, transform=transform), data

if __name__ == "__main__":
    data_dir = "/mnt/work/Code/credit_card_fd/Multi-GNN/data/Small_HI_split_trans"
    
    files = os.listdir(data_dir)
    files.sort()
    
    parser = create_parser()
    args = parser.parse_args()
    
    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)
    
    wandb.init(
        mode="disabled" if args.testing else "online",
        project="your_proj_name",

        config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
        }
    )

    config = wandb.config

    result_df = pd.DataFrame(columns=["Bank ID", "F1", "Balanced Accuracy", "Cohen's Kappa", "Recall", "Precision", "Length"])

    for f in tqdm.tqdm(files):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        file_path = os.path.join(data_dir, f)
        # print(file_path, args)
        # file_path = "/mnt/work/Code/credit_card_fd/Multi-GNN/data/Small_HI_split_trans/formatted_transactions.csv"
        loader, data = create_loader(file_path, args)
        
        preds = []
        ground_truths = []
        sample_batch = next(iter(loader))
        model = get_model(sample_batch, config, args)
        
        logging.info("=> loading model checkpoint")
        checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        # from IPython import embed 
        # embed()
        inds = torch.arange(data.edge_index.shape[1] + 1)
        f1, ba, cohens_kappa, recall, precision = evaluate_homo(loader, inds, model, data, device, args)
        new_row = pd.DataFrame({"Bank ID": [f], "F1": [f1], "Balanced Accuracy": [ba], "Cohen's Kappa": [cohens_kappa], "Recall": [recall], "Precision": [precision], "Length": [len(data.y)]})
        result_df = pd.concat([result_df, new_row], ignore_index=True)
        # break

    result_df.to_csv("results.csv", index=False)
