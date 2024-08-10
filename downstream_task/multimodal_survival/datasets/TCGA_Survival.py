import os
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
import torch.utils.data as data


class TCGA_Survival(data.Dataset):
    def __init__(self, excel_file, modal, root_path, root_omic, signatures=None):
        self.modal = modal
        self.signatures = signatures
        self.root_path = root_path
        self.root_omic = root_omic
        if self.signatures:
            csv = pd.read_csv(self.signatures)
            self.omic_names = []
            for col in csv.columns:
                omic = csv[col].dropna().unique()
                self.omic_names.append(omic)
        print("[dataset] loading dataset from %s" % (excel_file))
        rows = pd.read_csv(excel_file)
        self.rows = self.disc_label(rows)
        self.omics_size = self.len_omics(self.rows)
        self.path_size = self.len_path(self.rows)
        print("[dataset] sizes of omics data: ", self.omics_size)
        print("[dataset] dimension of WSI features: ", self.path_size)
        label_dist = self.rows["Label"].value_counts().sort_index()
        print("[dataset] discrete label distribution: ")
        print(label_dist)
        print("[dataset] required modality: %s" % (modal))
        for key in modal.split("_"):
            if key not in ["WSI", "Gene"]:
                raise NotImplementedError("modality [{}] is not implemented".format(modal))
        print("[dataset] dataset from %s, number of cases=%d" % (excel_file, len(self.rows)))

    def get_split(self):
        split = self.rows["split"].values.tolist()
        train_split = [i for i, x in enumerate(split) if x == "train"]
        val_split = [i for i, x in enumerate(split) if x == "val"]
        test_split = [i for i, x in enumerate(split) if x == "test"]
        val_split.extend(test_split)
        print("[dataset] training split: {}, validation split: {}".format(len(train_split), len(val_split)))
        return train_split, val_split

    def read_WSI(self, path):
        if str(path) == "nan":
            return torch.zeros((1))
        else:
            path = path.split(";")
            wsi = []
            for p in path:
                p = os.path.join(self.root_path, p.split("/")[-1])
                if os.path.exists(p):
                    wsi.append(torch.load(p))
                else:
                    print("missing file: ", p)
            wsi = torch.cat(wsi, dim=0).type(torch.float32)
            return wsi

    def read_Omics(self, path):
        if str(path) == "nan":
            return torch.zeros((1))
        else:
            path = os.path.join(self.root_omic, path.split("/")[-1])
            genes = pd.read_csv(path)
            if self.signatures:
                omic1 = torch.from_numpy(np.array(genes[genes["Gene"].isin(self.omic_names[0])]["Value"].values.tolist()).astype(np.float32))
                omic2 = torch.from_numpy(np.array(genes[genes["Gene"].isin(self.omic_names[1])]["Value"].values.tolist()).astype(np.float32))
                omic3 = torch.from_numpy(np.array(genes[genes["Gene"].isin(self.omic_names[2])]["Value"].values.tolist()).astype(np.float32))
                omic4 = torch.from_numpy(np.array(genes[genes["Gene"].isin(self.omic_names[3])]["Value"].values.tolist()).astype(np.float32))
                omic5 = torch.from_numpy(np.array(genes[genes["Gene"].isin(self.omic_names[4])]["Value"].values.tolist()).astype(np.float32))
                omic6 = torch.from_numpy(np.array(genes[genes["Gene"].isin(self.omic_names[5])]["Value"].values.tolist()).astype(np.float32))
                return (omic1, omic2, omic3, omic4, omic5, omic6)
            else:
                key = genes["Gene"].values.tolist()
                index = torch.from_numpy(np.array(genes["Index"].values.tolist()).astype(np.float32))
                value = torch.from_numpy(np.array(genes["Value"].values.tolist()).astype(np.float32))
                return (key, index, value)

    def __getitem__(self, index):
        case = self.rows.iloc[index, :].values.tolist()
        _, ID, Event, Status, WSI, RNA = case[:6]
        Label = case[-1]
        Censorship = 1 if int(Status) == 0 else 0
        if self.modal == "WSI":
            WSI = self.read_WSI(WSI)
            return (ID, WSI, Event, Censorship, Label)
        elif self.modal == "Gene":
            RNA = self.read_Omics(RNA)
            return (ID, RNA, Event, Censorship, Label)
        elif self.modal == "WSI_Gene":
            WSI = self.read_WSI(WSI)
            RNA = self.read_Omics(RNA)
            return (ID, WSI, RNA, Event, Censorship, Label)
        else:
            raise NotImplementedError("modality [{}] is not implemented".format(self.modal))

    def __len__(self):
        return len(self.rows)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows["Status"] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df["Event"], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows["Event"].max() + eps
        q_bins[0] = rows["Event"].min() - eps
        disc_labels, q_bins = pd.cut(rows["Event"], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), "Label", disc_labels)
        return rows

    def len_omics(self, rows):
        for omics in rows["RNA"].values.tolist():
            omics = os.path.join(self.root_omic, str(omics).split("/")[-1])
            if os.path.exists(omics):
                data = pd.read_csv(omics)
                if self.signatures:
                    omic1 = data[data["Gene"].isin(self.omic_names[0])]["Value"].values.tolist()
                    omic2 = data[data["Gene"].isin(self.omic_names[1])]["Value"].values.tolist()
                    omic3 = data[data["Gene"].isin(self.omic_names[2])]["Value"].values.tolist()
                    omic4 = data[data["Gene"].isin(self.omic_names[3])]["Value"].values.tolist()
                    omic5 = data[data["Gene"].isin(self.omic_names[4])]["Value"].values.tolist()
                    omic6 = data[data["Gene"].isin(self.omic_names[5])]["Value"].values.tolist()
                    return (len(omic1), len(omic2), len(omic3), len(omic4), len(omic5), len(omic6))
                else:
                    return len(data.iloc[:, 0].values.tolist())

    def len_path(self, rows):
        for wsi in rows["WSI"].values.tolist():
            wsi = os.path.join(self.root_path, str(wsi).split("/")[-1])
            if os.path.exists(wsi):
                data = torch.load(wsi)
                return data.size(-1)


if __name__ == "__main__":
    from torch.utils.data import DataLoader, SubsetRandomSampler

    dataset = TCGA_Survival(
        excel_file="/home/fzhouaf/codes/multimodal_for_foundation/csv/TCGA-All/BLCA_Splits.csv",
        modal="WSI_Gene",
        root_path="/project/llmponco/wangyihui/feature/TCGA-BLCA/pt_files/DinoV2_0_new",
        root_omic="/project/llmponco/wangyihui/Cbioportal/BLCA",
        signatures="/home/fzhouaf/codes/multimodal_for_foundation/csv/signatures.csv",
    )
    train_split, val_split = dataset.get_split(0, missing=False)
    train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(train_split))
    for batch_idx, (data_ID, data_WSI, data_Omics, data_Event, data_Censorship, data_Label) in enumerate(train_loader):
        data_Omics1 = data_Omics[0]
        data_Omics2 = data_Omics[1]
        data_Omics3 = data_Omics[2]
        data_Omics4 = data_Omics[3]
        data_Omics5 = data_Omics[4]
        data_Omics6 = data_Omics[5]
        print(data_ID, data_WSI.shape, data_Omics1.shape, data_Omics2.shape, data_Omics3.shape, data_Omics4.shape, data_Omics5.shape, data_Omics6.shape)
        # is there nan in data_WSI or data_Omics?
        if torch.isnan(data_WSI).any() or torch.isinf(data_WSI).any():
            print("nan in data_WSI")
            exit()
        if torch.isnan(data_Omics1).any() or torch.isinf(data_Omics1).any():
            print("nan in data_Omics1")
            exit()
        if torch.isnan(data_Omics2).any() or torch.isinf(data_Omics2).any():
            print("nan in data_Omics2")
            exit()
        if torch.isnan(data_Omics3).any() or torch.isinf(data_Omics3).any():
            print("nan in data_Omics3")
            exit()
        if torch.isnan(data_Omics4).any() or torch.isinf(data_Omics4).any():
            print("nan in data_Omics4")
            exit()
        if torch.isnan(data_Omics5).any() or torch.isinf(data_Omics5).any():
            print("nan in data_Omics5")
            exit()
        if torch.isnan(data_Omics6).any() or torch.isinf(data_Omics6).any():
            print("nan in data_Omics6")
            exit()
