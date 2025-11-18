import re
import torch
import numpy as np 
import h5py
from pathlib import Path

from torch.utils.data import Dataset, DataLoader


def get_paths(data_path, mode):
    data_path = Path(data_path)
    paths = np.array([x for x in data_path.iterdir()])
    paths = np.array([path for path in paths if re.findall("train" and ".h5", path.name)])
    paths = sorted(paths, key=lambda f: int("".join(filter(str.isdigit, str(f)))))
    return paths

#paths_train = get_paths("/hs/babbage/data/group-brueggen/David/kevin_data/", "train")[:2]
#paths_valid = get_paths("/workspace/transformer_data/", "valid")

class VisDataSet(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.len_data = len(paths)

    def __len__(self):
        return self.len_data

    def __getitem__(self, i):
        x, y = self.open_data(self.paths[i])
        return x, y

    def open_data(self, path):
        with h5py.File(str(path), "r") as file:
            vis = torch.from_numpy(np.array(file["vis_full"]))
            uvw = torch.from_numpy(np.array(file["uvw_full"]))
            lmn = torch.from_numpy(np.array(file["lmn"]))
            sky = torch.from_numpy(np.array(file["sky"]))
        return [vis, uvw, lmn], sky

#train_ds = VisDataSet(paths_train)
#validation_ds = VisDataSet(paths_valid)

#train_dl = DataLoader(train_ds, batch_size=2, shuffle=False)

#for x,y in train_dl:
#    vis = x[0]
#    uvw = x[1]
#    lmn = x[2]
#    print(vis.shape, uvw.shape, lmn.shape, y.shape)
#    print(uvw[0,0,:5])

