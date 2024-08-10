import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

class TCGA_Survival(data.Dataset):
    def __init__(self, csv_file, feature_path, modal, study, feature):
        self.modal = modal
        self.final_feature_path = os.path.join(feature_path, f'TCGA-{study}', 'pt_files', f'{feature}')
        print(f'feature path: {feature_path}')
        print(f'feature: {feature}')
        print(f'modal: {modal}')
        print(f'study: {study}')
        # for grouping gene sequence
        print('[dataset] loading dataset from %s' % (csv_file))
        rows = pd.read_csv(csv_file)
        self.rows = self.disc_label(rows)
        label_dist = self.rows['Label'].value_counts().sort_index()
        print('[dataset] discrete label distribution: ')
        print(label_dist)
        print('[dataset] dataset from %s, number of cases=%d' % (csv_file, len(self.rows)))

    def get_split(self):
        train_split = self.rows[self.rows['split'] == 'train'].index.tolist()
        val_split = self.rows[self.rows['split'] == 'validation'].index.tolist()
        test_split = self.rows[self.rows['split'] == 'test'].index.tolist()

        return train_split, val_split, test_split

    def read_WSI(self, WSI):
        wsi = [torch.load(os.path.join(self.final_feature_path, x)) for x in WSI.split(';')]
        wsi = torch.cat(wsi, dim=0)
        return wsi


    
    def __getitem__(self, index):
        case = self.rows.iloc[index, :].values.tolist()
        Study, ID, Event, Status, WSI = case[:5]
        Label = case[-1]
        Censorship = 1 if int(Status) == 0 else 0
        if self.modal == 'WSI':
            WSI = self.read_WSI(WSI)
            return (ID, WSI, Event, Censorship, Label)
        else:
            raise NotImplementedError('modality [{}] is not implemented'.format(self.modal))

    def __len__(self):
        return len(self.rows)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows['Status'] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows['Event'].max() + eps
        q_bins[0] = rows['Event'].min() - eps
        disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        # missing event data
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), 'Label', disc_labels)
        return rows
