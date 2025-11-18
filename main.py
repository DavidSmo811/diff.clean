import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb as wb
import os
import data_loader as dl

from unet_film2 import UNet
from major_loop import reconstruction_loop   # <-- falls in eigener Datei
# oder falls dein Code oben in reconstruction_loop.py gespeichert wird:
# from reconstruction_loop import reconstruction_loop


###############################################
#            CONFIGURATION
###############################################
RUN_NAME = "unet_reconstruction"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

num_max_major_cycle = 5
epochs_per_cycle = 200
learning_rate = 1e-4

batch_size = 5

###############################################


###############################################
#       YOUR DATASET – PLACEHOLDER
###############################################

data_path = dl.get_paths("/hs/babbage/data/shared/visibility_data/", "train")  # Passe den Pfad an

dataset = dl.VisDataSet(data_path)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

###############################################


###############################################
#              MODEL + OPTIMIZER
###############################################
unet = UNet(
    num_conditions=num_max_major_cycle + 1,
    in_channels=1,
    out_channels=1,
    base_c=32,
)

optimizer = optim.Adam(unet.parameters(), lr=learning_rate)
###############################################


###############################################
#              WEIGHTS & BIASES
###############################################
wb.init(project="major_cycle_unet",
        name=RUN_NAME,
        config={
            "max_major_cycles": num_max_major_cycle,
            "epochs_per_cycle": epochs_per_cycle,
            "learning_rate": learning_rate,
        })
###############################################


###############################################
#         START RECONSTRUCTION LOOP
###############################################
loop = reconstruction_loop(
    data_loader=data_loader,
    unet=unet,
    optimizer=optimizer,
    num_max_major_cycle=num_max_major_cycle,
    epochs=epochs_per_cycle,
)

# wichtig: Checkpoint-Verzeichnis übergeben
loop.checkpoint_dir = CHECKPOINT_DIR

print("Starting major-cycle training loop …")

for _ in loop:
    print("Major cycle step completed.")
    pass

print("Training finished.")