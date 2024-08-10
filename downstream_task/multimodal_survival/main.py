import os
import time

from datasets.TCGA_Survival import TCGA_Survival

from utils.options import parse_args
from utils.util import set_seed
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.util import Meter

from torch.utils.data import DataLoader, SubsetRandomSampler


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    if args.evaluate:
        results_dir = args.resume
    else:
        results_dir = "./results_{seed}/{dataset}/{model}/[{feature}]-[{time}]".format(
            seed=args.seed,
            dataset=args.excel_file.split("/")[-1].split(".")[0],
            model=args.model,
            feature=args.root_path.split("/")[-1],
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        )
    print("[checkpoint] results directory: ", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # define dataset
    dataset = TCGA_Survival(
        excel_file=args.excel_file,
        modal=args.modal,
        root_path=args.root_path,
        root_omic=args.root_omic,
        signatures=args.signatures,
    )
    args.num_classes = 4

    train_split, val_split = dataset.get_split()
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(train_split))
    val_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(val_split))

    # build model, criterion, optimizer, schedular
    meter = Meter()
    if args.model == "Porpoise":
        from models.Multimodal.Porpoise.network import Porpoise
        from models.Multimodal.Porpoise.engine import Engine

        model_dict = {"omic_input_dim": dataset.omics_size, "path_input_dim": dataset.path_size, "fusion": "lrb", "n_classes": args.num_classes}
        model = Porpoise(**model_dict)
        engine = Engine(args, results_dir)
    # Worked on grouped RNA squeeze
    elif args.model == "MCAT":
        from models.Multimodal.MCAT.network import MCAT_Surv
        from models.Multimodal.MCAT.engine import Engine

        model_dict = {"fusion": "concat", "path_size": dataset.path_size, "omic_sizes": dataset.omics_size, "n_classes": 4}
        model = MCAT_Surv(**model_dict)
        engine = Engine(args, results_dir)
    elif args.model == "CMTA":
        from models.Multimodal.CMTA.network import CMTA
        from models.Multimodal.CMTA.engine import Engine

        model_dict = {"path_size": dataset.path_size, "omic_sizes": dataset.omics_size, "n_classes": 4, "fusion": "concat", "model_size": "small"}
        model = CMTA(**model_dict)
        engine = Engine(args, results_dir)
    elif args.model == "MOTCAT":
        from models.Multimodal.MOTCAT.network import MOTCAT_Surv
        from models.Multimodal.MOTCAT.engine import Engine

        model_dict = {"path_size": dataset.path_size, "omic_sizes": dataset.omics_size, "n_classes": 4, "fusion": "concat"}
        model = MOTCAT_Surv(**model_dict)
        engine = Engine(args, results_dir)
    else:
        raise NotImplementedError("model [{}] is not implemented".format(args.model))

    print("[model] trained model: ", args.model)
    criterion = define_loss(args)
    print("[model] loss function: ", args.loss)
    optimizer = define_optimizer(args, model)
    print("[model] optimizer: ", args.optimizer)
    scheduler = define_scheduler(args, optimizer)
    print("[model] scheduler: ", args.scheduler)
    # start training
    if not args.evaluate:
        val_score, epoch = engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler)
        meter.updata(epoch, val_score)
        meter.save(os.path.join(results_dir, "results.csv"))
    else:
        engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler)


if __name__ == "__main__":

    args = parse_args()
    results = main(args)
    print("finished!")
