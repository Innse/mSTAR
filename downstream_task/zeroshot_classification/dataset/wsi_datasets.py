import os
from torch.utils.data import Dataset
import h5py
import numpy as np
import torch

def infer_folder_structure(source, subdir_name):
    """infer the folder structure of the dataset."""
    if not os.path.isdir(source):
        return 0
    if os.path.isdir(os.path.join(source, subdir_name)):
        depth = 2
    else:
        depth = 1
    return depth

class WSIEmbeddingDataset(Dataset):
    def __init__(self,
                 df,
                 target_transform=None,
                 index_col='slide_id',
                 target_col='label',
                 dir_path=None,
                 model_name = None,
                 label_map=None,
                 pid_slide_dict=None,
                 dummy_dim=0):
        """
        Args:
        """
        self.label_map = label_map
        self.index_col = index_col
        self.target_col = target_col
        self.dir_path = dir_path
        self.model_name = model_name
        self.target_transform = target_transform
        self.data = df
        
        # use h5 format or pt format

        # pid_slide_dict is a dictionary that maps patient id to slide ids
        self.pid_slide_dict = pid_slide_dict

        # dummy_dim is the dimension of the dummy feature vector, if dummy_dim > 0
        self.dummy_dim = dummy_dim

    def __len__(self):
        return len(self.data)

    def get_ids(self, ids):
        return self.data.loc[ids, self.index_col]

    def get_labels(self, ids):
        if self.target_col is None:
            return []
        return self.data.loc[ids, self.target_col]
    
    def get_feat_path(self, dir_path, model,slide_id):

        feat_path = os.path.join(dir_path, 'pt_files', model, slide_id + '.pt')
        return feat_path
   
    def __getitem__(self, idx):
        slide_id = self.get_ids(idx)
        label = self.get_labels(idx)
        dir_path = self.dir_path

        if self.pid_slide_dict is not None:
            slide_ids = self.pid_slide_dict[slide_id]
        else:
            slide_ids = [slide_id]

        if self.label_map is not None:
            label = self.label_map[label]
        if self.target_transform is not None:
            label = self.target_transform(label)

        all_features = []
        all_coords = []
        
        if self.dummy_dim > 0:
            all_features.append(np.zeros((1000, self.dummy_dim)))
        else:
            for slide_id in slide_ids:
                feat_path = self.get_feat_path(dir_path, self.model_name, slide_id)
                features = torch.load(feat_path)
                all_features.append(features)

        all_features = torch.from_numpy(np.concatenate(all_features, axis=0))

        if len(all_features.size()) == 3:
            all_features = all_features.squeeze(0)

        if len(all_coords) > 0:
            all_coords = np.concatenate(all_coords, axis=0)
            if len(all_coords.shape) == 3:
                all_coords = all_coords.squeeze(0)
        
        out = {'img': all_features, 'label': label, 'coords': all_coords, 'slide_id': slide_id}
        return out