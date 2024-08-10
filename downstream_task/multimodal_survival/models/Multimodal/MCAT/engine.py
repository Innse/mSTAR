import os
import numpy as np
from tqdm import tqdm
from typing import Dict

from sksurv.metrics import concordance_index_censored

import torch.optim
from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers.bootstrapping import BootStrapper


class Engine(object):
    def __init__(self, args, results_dir):
        self.args = args
        self.results_dir = results_dir
        # tensorboard
        if args.log_data:
            from tensorboardX import SummaryWriter

            self.writer = SummaryWriter(self.results_dir, flush_secs=15)
        self.best_scores = None
        self.best_epoch = 0
        self.filename_best = None
        self.meter = self.metric(bootstrap=True)

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()
        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_scores = checkpoint["best_scores"]
                self.best_epoch = checkpoint["best_epoch"]
                model.load_state_dict(checkpoint["state_dict"])
                print("=> loaded checkpoint (score: {})".format(checkpoint["best_score"]))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            # evaluate on validation set
            scores = self.validate(val_loader, model, criterion)
            # remember best c-index and save checkpoint
            is_best = scores["mean"] > self.best_scores["mean"] if self.best_scores is not None else True
            if is_best:
                self.best_scores = scores
                self.best_epoch = self.epoch
                self.save_checkpoint({"best_epoch": epoch, "state_dict": model.state_dict(), "best_score": self.best_scores})
            print(" *** best score={:.4f} at epoch {}".format(self.best_scores["mean"], self.best_epoch))
            scheduler.step()
            print(">>>")
            print(">>>")
        return self.best_scores, self.best_epoch

    def train(self, data_loader, model, criterion, optimizer):
        model.train()

        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        if self.args.tqdm:
            dataloader = tqdm(data_loader, desc="Train Epoch {}".format(self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------train epoch {}-------------------------------".format(self.epoch))

        for batch_idx, (data_ID, data_WSI, data_Omics, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            data_Omics1 = data_Omics[0]
            data_Omics2 = data_Omics[1]
            data_Omics3 = data_Omics[2]
            data_Omics4 = data_Omics[3]
            data_Omics5 = data_Omics[4]
            data_Omics6 = data_Omics[5]
            if torch.cuda.is_available():
                data_WSI = data_WSI.type(torch.FloatTensor).cuda()
                data_Omics1 = data_Omics1.type(torch.FloatTensor).cuda()
                data_Omics2 = data_Omics2.type(torch.FloatTensor).cuda()
                data_Omics3 = data_Omics3.type(torch.FloatTensor).cuda()
                data_Omics4 = data_Omics4.type(torch.FloatTensor).cuda()
                data_Omics5 = data_Omics5.type(torch.FloatTensor).cuda()
                data_Omics6 = data_Omics6.type(torch.FloatTensor).cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            # prediction
            hazards, S, _, _ = model(x_path=data_WSI, x_omic1=data_Omics1, x_omic2=data_Omics2, x_omic3=data_Omics3, x_omic4=data_Omics4, x_omic5=data_Omics5, x_omic6=data_Omics6)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
            total_loss += loss.item()
            # backward to update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        all_risk_scores = torch.from_numpy(all_risk_scores)
        all_censorships = torch.from_numpy(all_censorships)
        all_event_times = torch.from_numpy(all_event_times)
        self.meter.update((1 - all_censorships).to(torch.bool), all_event_times, all_risk_scores)
        scores = self.meter.compute()
        print("loss: {:.4f}, c_index: {:.4f}±{:.4f}".format(loss, scores["mean"], scores["std"]))
        if self.writer:
            self.writer.add_scalar("train/loss", loss, self.epoch)
            self.writer.add_scalar("train/c_index_mean", scores["mean"], self.epoch)
            self.writer.add_scalar("train/c_index_std", scores["std"], self.epoch)

    def validate(self, data_loader, model, criterion, split="val"):
        model.eval()
        total_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        if self.args.tqdm:
            dataloader = tqdm(data_loader, desc="{} epoch {}".format(split, self.epoch))
        else:
            dataloader = data_loader
            print("-------------------------------{} epoch {}-------------------------------".format(split, self.epoch))

        for batch_idx, (data_ID, data_WSI, data_Omics, data_Event, data_Censorship, data_Label) in enumerate(dataloader):
            data_Omics1 = data_Omics[0]
            data_Omics2 = data_Omics[1]
            data_Omics3 = data_Omics[2]
            data_Omics4 = data_Omics[3]
            data_Omics5 = data_Omics[4]
            data_Omics6 = data_Omics[5]
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_Omics1 = data_Omics1.type(torch.FloatTensor).cuda()
                data_Omics2 = data_Omics2.type(torch.FloatTensor).cuda()
                data_Omics3 = data_Omics3.type(torch.FloatTensor).cuda()
                data_Omics4 = data_Omics4.type(torch.FloatTensor).cuda()
                data_Omics5 = data_Omics5.type(torch.FloatTensor).cuda()
                data_Omics6 = data_Omics6.type(torch.FloatTensor).cuda()
                data_Label = data_Label.type(torch.LongTensor).cuda()
                data_Censorship = data_Censorship.type(torch.FloatTensor).cuda()
            # prediction
            with torch.no_grad():
                hazards, S, _, _ = model(x_path=data_WSI, x_omic1=data_Omics1, x_omic2=data_Omics2, x_omic3=data_Omics3, x_omic4=data_Omics4, x_omic5=data_Omics5, x_omic6=data_Omics6)
            loss = criterion(hazards=hazards, S=S, Y=data_Label, c=data_Censorship)
            total_loss += loss.item()
            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = data_Censorship.item()
            all_event_times[batch_idx] = data_Event
        # calculate loss and error for each epoch
        loss = total_loss / len(dataloader)
        all_risk_scores = torch.from_numpy(all_risk_scores)
        all_censorships = torch.from_numpy(all_censorships)
        all_event_times = torch.from_numpy(all_event_times)
        self.meter.update((1 - all_censorships).to(torch.bool), all_event_times, all_risk_scores)
        scores = self.meter.compute()
        print("loss: {:.4f}, c_index: {:.4f}±{:.4f}".format(loss, scores["mean"], scores["std"]))
        if self.writer:
            self.writer.add_scalar("val/loss", loss, self.epoch)
            self.writer.add_scalar("val/c_index_mean", scores["mean"], self.epoch)
            self.writer.add_scalar("val/c_index_std", scores["std"], self.epoch)
        return scores

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(
            self.results_dir,
            "model_best_{val_score:.4f}_{epoch}.pth.tar".format(
                val_score=self.best_scores["mean"],
                epoch=self.best_epoch,
            ),
        )
        print("save best model {filename}".format(filename=self.filename_best))
        torch.save(state, self.filename_best)

    def metric(self, bootstrap=False):
        metrics: Dict[str, Metric] = {
            "C-index": ConcordanceIndexCensored(),
        }
        # boot strap wrap
        if bootstrap:
            for k, m in metrics.items():
                print("wrapping:", k)
                metrics[k] = BootStrapper(m, num_bootstraps=1000, sampling_strategy="multinomial")
        metrics = MetricCollection(metrics)
        return metrics


class ConcordanceIndexCensored(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("censorships", default=[], dist_reduce_fx=None)
        self.add_state("event_times", default=[], dist_reduce_fx=None)
        self.add_state("risk_scores", default=[], dist_reduce_fx=None)

    def update(self, censorships, event_times, risk_scores):
        # Concatenate new predictions, targets, and event indicators to the existing lists
        self.censorships = censorships
        self.event_times = event_times
        self.risk_scores = risk_scores

    def compute(self):
        c_index = concordance_index_censored(self.censorships, self.event_times, self.risk_scores, tied_tol=1e-08)[0]
        c_index = torch.tensor(c_index, dtype=torch.float32)
        self.reset()
        return c_index

    def reset(self):
        # Reset the state by clearing the lists
        self.preds = []
        self.targets = []
        self.event_indicator = []
