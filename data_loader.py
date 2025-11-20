import re
import torch
import numpy as np 
import h5py
from pathlib import Path
import webdataset as wds
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


class BufferedVisDataSet(torch.utils.data.Dataset):
    def __init__(self, all_paths, preload_size=500):
        self.all_paths = all_paths
        self.preload_size = preload_size
        self.buffer = []
        self.start_index = 0
        self._load_buffer()

    def _load_buffer(self):
        print(f"Loading next {self.preload_size} samples into RAM...")
        self.buffer = []

        for p in self.all_paths[self.start_index : self.start_index + self.preload_size]:
            with h5py.File(p, "r") as f:
                vis = torch.from_numpy(np.array(f["vis_full"]))
                uvw = torch.from_numpy(np.array(f["uvw_full"]))
                lmn = torch.from_numpy(np.array(f["lmn"]))
                sky = torch.from_numpy(np.array(f["sky"]))
            self.buffer.append(([vis, uvw, lmn], sky))

        self.start_index += self.preload_size

    def __len__(self):
        return len(self.buffer)

    def refill(self):
        """Call this when the buffer is exhausted."""
        if self.start_index < len(self.all_paths):
            self._load_buffer()

    def __getitem__(self, idx):
        return self.buffer[idx]
    



# def make_webdataset_dataloader(
#     shards_pattern, 
#     batch_size=4,
#     shuffle=1,
#     num_workers=4,
# ):
#     """
#     Returns a PyTorch DataLoader that yields batches of:
#         (vis, uvw, lmn), sky
#     """

#     # WebDataset pipeline
#     dataset = (
#         wds.WebDataset(shards_pattern, shardshuffle=True)
#         .shuffle(shuffle)
#         .decode()  
#         .to_tuple("vis.npy", "uvw.npy", "lmn.npy", "sky.npy")
#         .map(lambda x: (
#             (torch.from_numpy(x[0]),
#              torch.from_numpy(x[1]),
#              torch.from_numpy(x[2])),
#             torch.from_numpy(x[3]),
#         ))
#         .batched(batch_size, partial=False)
#     )

#     # PyTorch DataLoader wrapper
#     loader = DataLoader(
#         dataset,
#         batch_size=None,   # <--- important! batches already created by dataset
#         num_workers=num_workers,
#     )

#     return loader
def make_webdataset_dataloader(
    shards_pattern,
    batch_size=4,
    shuffle=1,
    num_workers=4,
):
    dataset = (
        wds.WebDataset(shards_pattern, shardshuffle=shuffle)
        .shuffle(shuffle)
        .decode("torch")
        .to_tuple("vis.npy", "uvw.npy", "lmn.npy", "sky.npy")
        .map(lambda x: (
            (torch.from_numpy(x[0]) if isinstance(x[0], np.ndarray) else x[0],
             torch.from_numpy(x[1]) if isinstance(x[1], np.ndarray) else x[1],
             torch.from_numpy(x[2]) if isinstance(x[2], np.ndarray) else x[2]),
            torch.from_numpy(x[3]) if isinstance(x[3], np.ndarray) else x[3]
        ))
    )

    # Verwende PyTorch's eigenen DataLoader mit collate_fn
    def collate_fn(batch):
        vis = torch.stack([x[0][0] for x in batch])
        uvw = torch.stack([x[0][1] for x in batch])
        lmn = torch.stack([x[0][2] for x in batch])
        sky = torch.stack([x[1] for x in batch])
        return (vis, uvw, lmn), sky

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader