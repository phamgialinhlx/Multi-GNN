import time
import logging
from util import create_parser, set_seed, logger_setup
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_homo, evaluate_hetero
from torch_geometric.nn import to_hetero, summary
from data_loading import get_data
from training import get_model
import json
import torch
import torch.utils.benchmark as benchmark

def main():
    parser = create_parser()
    args = parser.parse_args()

    with open('data_config.json', 'r') as config_file:
        data_config = json.load(config_file)

    # Setup logging
    logger_setup()

    #set seed
    set_seed(args.seed)

    #get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()
    
    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)
    
    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None
    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)
    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')
    logging.info("=> loading model checkpoint")
    checkpoint = torch.load(f'{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar')
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    logging.info("=> loaded checkpoint (epoch {})".format(start_epoch))
    
    # Benchmark throughput
    num_runs = 100  # Number of runs for benchmarking
    
    # Set up input tensors
    input_tensors = (sample_batch.x, sample_batch.edge_index, sample_batch.edge_attr)
    
    # Create benchmark input
    benchmark_input = (model, input_tensors)
    
    # Run benchmark
    t = benchmark.Timer(
        stmt='model(*input_tensors)',
        setup='pass',
        globals={'model': model, 'input_tensors': input_tensors}
    )
    print(t.timeit(num_runs))
if __name__ == "__main__":
    main()
