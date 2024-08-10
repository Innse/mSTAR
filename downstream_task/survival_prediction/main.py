import os
import time
import sys

from datasets.TCGA_Survival import TCGA_Survival

from utils.options import parse_args
from utils.util import set_seed
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import wandb

def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    if args.evaluate:
        results_dir = args.resume
    else:
        results_dir = "./results/{modal}/{dataset}/[{model}-{feature}]-[{time}]".format(
            modal=args.modal,
            dataset=args.csv_file.split('/')[-1].split('.')[0],
            model=args.model,
            feature=args.feature,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S")
        )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # define dataset
    wandb.init(project=args.csv_file.split('/')[-1].split('.')[0].split('_')[0]+'_survival', config = args)
    dataset = TCGA_Survival(csv_file=args.csv_file, feature_path=args.feature_path, modal=args.modal, study=args.study, feature=args.feature)
    args.num_classes = 4
    if args.feature in ['plip','conch']:
        args.n_features = 512
    else:
        args.n_features = 1024

        # get split
    train_split, val_split, test_split = dataset.get_split()
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(train_split))
    val_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(val_split+test_split))
    test_subset = Subset(dataset, val_split+test_split)
    # build model, criterion, optimizer, schedular
    #################################################
    if args.model == "AttMIL":
        from models.AttMIL.network import DAttention
        from models.AttMIL.engine import Engine
        model = DAttention(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
        engine = Engine(args, results_dir)
    elif args.model == "TransMIL":
        from models.TransMIL.network import TransMIL
        from models.TransMIL.engine import Engine
        model = TransMIL(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
        engine = Engine(args, results_dir)
    elif args.model == "TransMIL_Pre":
        from models.TransMIL_Pre.network import TransMIL
        from models.TransMIL_Pre.engine import Engine
        
        assert os.path.exists(args.aggregator), "aggregator checkpoint not found at {}".format(args.aggregator)
        checkpoint = torch.load(args.aggregator ,map_location='cpu')
        
        model = TransMIL(n_classes=args.num_classes, dropout=0.25, act="relu", n_features=args.n_features)
        engine = Engine(args, results_dir)
        
        vision_encoder_params = {
            k.replace('module.', '').replace('vision_encoder.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if 'vision_encoder' in k
        }

        load_result = model.load_state_dict(vision_encoder_params, strict=False)

        # Check if there are any missing or unexpected keys
        if load_result.missing_keys:
            print("Warning: Missing keys detected during the loading of aggregator parameters:", load_result.missing_keys)
        if load_result.unexpected_keys:
            print("Warning: Unexpected keys detected during the loading of aggregator parameters:", load_result.unexpected_keys)

        # If there are neither missing nor unexpected keys, print a success message
        if not load_result.missing_keys and not load_result.unexpected_keys:
            print("Success: aggregator parameters loaded correctly into the model.")

        engine = Engine(args, results_dir)
    else:
        raise NotImplementedError("model [{}] is not implemented".format(args.model))
    print('[model] trained model: ', args.model)
    criterion = define_loss(args)
    print('[model] loss function: ', args.loss)
    optimizer = define_optimizer(args, model)
    print('[model] optimizer: ', args.optimizer)
    scheduler = define_scheduler(args, optimizer)
    print('[model] scheduler: ', args.scheduler)
    # start training
    mean_c_index, ci_lower, ci_upper = engine.learning(model, train_loader, val_loader, test_subset, criterion, optimizer, scheduler)
    
    cindex_summary = f"{mean_c_index:.4f}({ci_lower:.4f},{ci_upper:.4f})"
    wandb.log({"C-index Summary": cindex_summary})


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
