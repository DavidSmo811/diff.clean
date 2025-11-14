import h5py
import os, logging
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from models.model import Minorloop
from models.network import Network
from models.loss import mse_loss
from models.metric import mae
from core.logger import VisualWriter, InfoLogger
import numpy as np



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

opt = {
    'datasets': {
        'train': {
            'dataloader': {
                'args': {
                    'batch_size': 4
                }
            }
        },
        'test': {
            'dataloader': {
                'args': {
                    'batch_size': 1
                }
            }
        }
    },
    'phase': 'train',                # oder 'test'
    'distributed': False,
    'gpu_ids': [0],
    'seed': 42,
    'global_rank': 0,
    'train': {
        'n_epoch': 100,
        'n_iter': 1000000,
        'log_iter': 100,
        'val_epoch': 1,
        'save_checkpoint_epoch': 10,
        'tensorboard': True,
    },
    'path': {
        'experiments_root': '/hs/babbage/data/group-brueggen/David/diffusion/MinorLoop/experiments/sky_reconstruction',
        'results': 'results',
        'checkpoint': 'checkpoint',
        'resume_state': None
    },
}


class NpySkyDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.gt_files = sorted([f for f in os.listdir(root) if f.endswith("_gt.npy")])

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, idx):
        gt_path = os.path.join(self.root, self.gt_files[idx])
        cond_path = gt_path.replace("_gt.npy", "_cond.npy")

        gt_image = np.load(gt_path)
        cond_image = np.load(cond_path).reshape(512, 512)

        # Normierung auf [-1, 1]
        gt_image = (gt_image - gt_image.min()) / (gt_image.max() - gt_image.min()) * 2 - 1
        cond_image = (cond_image - cond_image.min()) / (cond_image.max() - cond_image.min()) * 2 - 1

        gt_image = torch.abs(torch.tensor(gt_image).unsqueeze(0))   # (1, H, W)
        cond_image = torch.abs(torch.tensor(cond_image).unsqueeze(0)) # (1, H, W)



        return {
            "cond_image": cond_image,
            "gt_image": gt_image,
            "path": os.path.basename(gt_path)
        }

log_dir = opt['path']['experiments_root']
os.makedirs(log_dir, exist_ok=True)
info_logger = InfoLogger(opt)
logger = info_logger.logger

logger.setLevel(logging.INFO)

# FileHandler hinzufügen
fh = logging.FileHandler(os.path.join(log_dir, f"{opt['phase']}.log"))
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

opt['path']['tb_logger'] = '/hs/babbage/data/group-brueggen/David/diffusion/MinorLoop/logs'
writer = VisualWriter(opt, logger=logger)

train_dataset = NpySkyDataset("/hs/babbage/data/group-brueggen/David/kevin_data/")
val_dataset = NpySkyDataset("/hs/babbage/data/group-brueggen/David/kevin_data/subsample_val/")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)


network_args = {
    "init_type": "kaiming",
    #"module_name": "guided_diffusion",
    "unet": {
        "in_channel": 2,          # gray input + conditional image
        "out_channel": 1,         # gray restored output
        "inner_channel": 64,
        "channel_mults": [1, 2, 4, 8],
        "attn_res": [16],
        "num_head_channels": 32,
        "res_blocks": 2,
        "dropout": 0.2,
        "image_size": 512
    },
    "beta_schedule": {
        "train": {
            "schedule": "linear",
            "n_timestep": 2000,
            "linear_start": 1e-6,
            "linear_end": 0.01
        },
        "test": {
            "schedule": "linear",
            "n_timestep": 1000,
            "linear_start": 1e-4,
            "linear_end": 0.09
        }
    }
}

netG = Network(**network_args).to(device)


losses = [mse_loss]
metrics = [mae]
optimizers = [{"lr": 5e-5, "weight_decay": 0}]

# EMA (optional, aber wie in JSON)
ema_scheduler = {
    "ema_start": 1,
    "ema_iter": 1,
    "ema_decay": 0.9999
}


model = Minorloop(
    networks=[netG],
    losses=losses,
    sample_num=8,
    task="restoration",
    optimizers=optimizers,
    ema_scheduler=ema_scheduler,
    opt=opt,
    phase_loader=train_loader,
    val_loader=val_loader,
    metrics=metrics,
    logger=logger,
    writer=writer
)


if opt['phase'] == 'train':
    model.train()
else:
    model.test()