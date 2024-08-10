import os
import csv
import random
import numpy as np

import torch


def set_seed(seed=7):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Meter:
    def __init__(self):
        self.header = ["epoch", "c-index"]
        self.results = []

    def updata(self, epoch, val_score):
        val_score = {k: v.item() for k, v in val_score.items()}
        # convert the tensor to float
        self.results.append(epoch)
        self.results.append(str(round(val_score["mean"], 4)) + "," + str(round(val_score["std"], 4)))

    def save(self, path):
        print("save evaluation resluts to", path)
        with open(path, "a", encoding="utf-8-sig", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(self.header)
            writer.writerow(self.results)
