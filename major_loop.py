import subprocess

def get_least_used_gpu():
    output = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
        shell=True
    )
    mem_usage = [int(x) for x in output.decode().strip().split("\n")]
    return min(range(len(mem_usage)), key=lambda i: mem_usage[i])

gpu_id = get_least_used_gpu()
print(f"Selected GPU: {gpu_id}")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cupy as cp
import os
import wandb as wb
from joblib import Parallel, delayed
from finufft_dlpack import CupyFinufft
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import io
import subprocess
from PIL import Image

import loss as ls

# def get_least_used_gpu():
#     """Returns the index of the least memory-used GPU."""
#     try:
#         result = subprocess.check_output(
#             "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
#             shell=True
#         )
#         gpu_mem = [int(x) for x in result.decode("utf-8").strip().split("\n")]
#         return int(torch.argmin(torch.tensor(gpu_mem)))
#     except Exception as e:
#         print("Could not determine free GPU, defaulting to cuda:0. Error:", e)
#         return 0

def wandb_powernorm_image(array, caption="", gamma=0.5, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)

    im = ax.imshow(array, cmap=cmap, norm=PowerNorm(gamma))
    plt.colorbar(im, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    pil_img = Image.open(buf)

    return wb.Image(pil_img, caption=caption)


class reconstruction_loop:
    def __init__(self, data_loader, unet, optimizer, optimizer_major ,num_max_major_cycle, epochs, pix_size=512, fov_arcsec=6000, eps=1e-6, max_resmean=1e-3, device=None, SFED=390, noise_factor=0, chan_width=300000, exposure=16):
        self.dataloader = data_loader
        #self.dataset = data_set
        self.unet = unet
        self.optimizer = optimizer
        self.optimizer_major = optimizer_major
        self.finufft = CupyFinufft(image_size=pix_size, fov_arcsec=fov_arcsec, eps=eps)
        self.pix_size = pix_size
        self.num_max_major_cycle = num_max_major_cycle
        self.model_image = None
        self.model_vis = None
        self.max_resmean = max_resmean
        self.epochs = epochs
        self.resmean = -1
        self.current_epoch = 0
        self.current_cycle = 0
        self.SFED=SFED
        self.noise_factor=noise_factor
        self.chan_width=chan_width
        self.exposure=exposure

        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # if device is not None:
        #     self.device = device
        # else:
        #     if torch.cuda.is_available():
        #         gpu_id = get_least_used_gpu()
        #         self.device = torch.device(f"cuda:{gpu_id}")
        #         print(f"Using GPU {gpu_id} (least memory used)")
        #     else:
        #         self.device = torch.device("cpu")
        self.unet.to(self.device)
        self.loss_fn = ls.AdaptiveRadioReconstructionLoss(device=self.device)
        self.data_iter = None
        self.batch_count = 0

    def save_checkpoint(self, name):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(self.unet.state_dict(), path)
        print(f"[Checkpoint] Saved: {path}")


    def generate_noise(self, shape):
        # scaling factor for the noise
        factor = 20#1

        # system efficency factor, near 1
        eta = 0.93

        # taken from simulations
        chan_width = torch.tensor([self.chan_width])#obs.bandwidths[0] * len(obs.bandwidths)#obs.bandwith ist im Fall von Meerkat 214000

        # corr_int_time
        exposure = self.exposure #obs.int_time

        # taken from:
        # https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/sensitivity

        std = factor * 1 / eta * self.SEFD
        std /= torch.sqrt(2 * exposure * chan_width)
        print("STD",std)
        std=std.item()
        noise = torch.normal(mean=0, std=std, size=shape, device=self.device)
        noise = noise + 1.0j * torch.normal(mean=0, std=std, size=shape, device=self.device)

        return noise

    def __iter__(self):
        self.current_cycle = 0
        self.current_epoch = 0
        self.model_vis = None
        self.model_image = None
        self.data_iter = None
        self.batch_count = 0
        return self
    
    # def __next__(self):
    #     if self.current_cycle > self.num_max_major_cycle:
    #         print("Reached maximum number of major cycles.")
    #         self.save_checkpoint(f"unet_final_cycle_{self.current_cycle}.pt")
    #         raise StopIteration
    #     if self.resmean <= self.max_resmean and self.resmean != -1:
    #         print(f"Stopping at major cycle {self.current_cycle} with residual mean {self.resmean}")
    #         self.save_checkpoint(f"unet_converged_cycle_{self.current_cycle}.pt")
    #         raise StopIteration
    #     self.current_epoch +=1
    #     self.model_image = None
    #     if self.current_epoch > self.epochs:
    #         self.current_cycle += 1
    #         self.current_epoch = 0
    #         self.save_checkpoint(f"unet_cycle_{self.current_cycle}.pt")
    #     return self.major_cycle_step(self.dataloader)


    # def __next__(self):
    #     print("Bin in next")
    #     # Abbruchbedingungen prüfen
    #     if self.current_cycle > self.num_max_major_cycle:
    #         print("Reached maximum number of major cycles.")
    #         self.save_checkpoint(f"unet_final_cycle_{self.current_cycle}.pt")
    #         raise StopIteration
    #     if self.resmean <= self.max_resmean and self.resmean != -1:
    #         print(f"Stopping at major cycle {self.current_cycle} with residual mean {self.resmean}")
    #         self.save_checkpoint(f"unet_converged_cycle_{self.current_cycle}.pt")
    #         raise StopIteration

    #     # Major Cycle Schritt ausführen
    #     result = self.major_cycle_step(self.dataloader)

    #     # Epoche erhöhen
    #     self.current_epoch += 1
    #     self.model_image = None 

    #     # Prüfen, ob ein neuer Major Cycle beginnt
    #     if self.current_epoch > self.epochs:
    #         self.current_cycle += 1
    #         self.current_epoch = 0
    #         self.save_checkpoint(f"unet_cycle_{self.current_cycle}.pt")

    #     return result
    #@profile
    def __next__(self):
        print(f"In __next__: Cycle {self.current_cycle}, Epoch {self.current_epoch}")
        self.current_epoch += 1
        self.model_image = None
        if self.current_epoch >= self.epochs:
            print(f"  → Major Cycle {self.current_cycle} completed")
            self.save_checkpoint(f"unet_cycle_{self.current_cycle}.pt")
            self.current_cycle += 1
            self.current_epoch = 0
            self.model_image = None
            #self.dataset.refill()
        
        # Abbruchbedingungen prüfen
        if self.current_cycle >= self.num_max_major_cycle:
            print("Reached maximum number of major cycles.")
            self.save_checkpoint(f"unet_final_cycle_{self.current_cycle}.pt")
            raise StopIteration
        
        if self.resmean <= self.max_resmean and self.resmean != -1:
            print(f"Stopping at major cycle {self.current_cycle} with residual mean {self.resmean}")
            self.save_checkpoint(f"unet_converged_cycle_{self.current_cycle}.pt")
            raise StopIteration

        # DataLoader-Iterator initialisieren oder neu starten
        if self.data_iter is None:
            print(f"  → Starting new epoch {self.current_epoch}")
            self.data_iter = iter(self.dataloader)
            self.batch_count = 0
        
        try:
            # Nächsten Batch holen
            x, y = next(self.data_iter)
            self.batch_count += 1
            print(f"  → Processing batch {self.batch_count}")
            
            # Major cycle step mit dem aktuellen Batch
            result = self.major_cycle_step(x, y)
            
            return result
            
        except StopIteration:
            # DataLoader ist durch → Epoch fertig
            print(f"  → Epoch {self.current_epoch} finished")
            self.data_iter = None
            self.current_epoch += 1
            
            # Nach erster Epoch: model_image zurücksetzen
            if self.current_epoch == 1:
                print(f"  → Resetting model_image after first epoch")
                self.model_image = None
            
        # Prüfen, ob Major Cycle fertig ist
        if self.current_epoch >= self.epochs:
            print(f"  → Major Cycle {self.current_cycle} completed")
            self.save_checkpoint(f"unet_cycle_{self.current_cycle}.pt")
            self.current_cycle += 1
            self.current_epoch = 0
            self.model_image = None
            
            # Rekursiv nächsten Batch holen (neue Epoch startet)
            return self.__next__()

    #@profile
    def loss_percentile_scheduler(self, ground_truth, start_percentile=99.95, end_percentile=70.0, k=5.0, softness=0.1):
        x = torch.linspace(0, 1, self.num_max_major_cycle, dtype=ground_truth.dtype).to(self.device)  # linear 0..1
        k = 5.0  # >1 → starkes Anhäufen am Ende
        y = start_percentile + (end_percentile - start_percentile) * ((1 - x)**k)
        y=torch.flip(y, dims=[0])
        print(ground_truth.dtype)
        print("Alle Percentile",y)
        print("Percentil",y[self.current_cycle])
        percentile_gt = torch.quantile(torch.abs(ground_truth), y[self.current_cycle] / 100.0)
        #filter_image = torch.where(torch.abs(ground_truth) >= percentile_gt, ground_truth, torch.zeros_like(ground_truth))

        mask = torch.sigmoid((ground_truth - percentile_gt) / (softness * percentile_gt))

        filter_image = mask * ground_truth

        return filter_image

    
    #@profile
    def minor_cycle_step(self, residual_image, sky_image, train_cycle, logging_step=2):
        print(self.current_cycle, self.current_epoch)
        if residual_image.dim() == 3:
            residual_image = residual_image.unsqueeze(1)  # (B,1,H,W)

        if self.model_image.dim() == 3:
            self.model_image = self.model_image.unsqueeze(1)  # (B,1,H,W)

        print("Vorher",sky_image.shape)
        if sky_image.dim() == 3:
            
            sky_image = sky_image.unsqueeze(1)
            print("Nachher",sky_image.shape)

        train = True #train_cycle==self.current_cycle
        if train:
            loss_image=sky_image # self.loss_percentile_scheduler(sky_image)
            loss_image=loss_image.float()
            self.unet.train()
            self.optimizer.zero_grad()
            input_image = torch.cat([residual_image.detach(), self.model_image.detach()], dim=1)
            print("Input image shape:", input_image.shape)
            print("Self.model_image shape:", self.model_image.shape)
            #input_image = input_image/torch.max(torch.abs(self.dirty_image)).detach()#residual_image.detach()/torch.max(torch.abs(self.dirty_image.detach()))
            prediction = self.unet(input_image, conditioning=train_cycle)
            #prediction = prediction * torch.max(torch.abs(self.dirty_image)).detach()
            print("Prediction shape:", prediction.shape)

            #Because the network now also receives the model image as input, the network can correct its previous prediction and we can take the absolute value here
            prediction = torch.abs(prediction)

            self.model_image = prediction #torch.abs(self.model_image.detach() + prediction if self.model_image is not None else prediction)
            #Vielleicht doch einfach self.model_image = prediction nehmen? Dann kann das Netzwerk Fehler korrigieren!
            print("Model image shape:", self.model_image.shape)
            #loss1 = nn.HuberLoss()(torch.log(torch.abs(self.model_image)+1e-7), torch.log(torch.abs(loss_image)+1e-7))/nn.HuberLoss()(torch.log(torch.abs(self.dirty_image)+1e-7), torch.log(torch.abs(loss_image)+1e-7))
            #percentile_gt = torch.quantile(torch.abs(loss_image), 99.95 / 100.0)
            #mask = torch.where(torch.abs(loss_image) >= percentile_gt, 1, torch.zeros_like(loss_image))#(loss_image != 0).float()
            #loss2 = nn.L1Loss()(self.model_image*mask, loss_image*mask)/nn.L1Loss()(self.dirty_image*mask, loss_image*mask)
            #loss = 0.3*loss1 + 0.7*loss2
            loss, loss_dict = self.loss_fn(pred=self.model_image, target=loss_image, dirty=self.dirty_image, epoch=train_cycle, total_epochs=self.num_max_major_cycle)
            loss.backward()
            self.optimizer.step()
            wb.log({f"minor_cycle_loss {train_cycle}": loss.item(), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
            #if self.current_epoch % logging_step == 0:
            wb.log({"residual image": wandb_powernorm_image(residual_image[0,0].cpu().detach().numpy(), caption="Residual Image"), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
            wb.log({"model image": wandb_powernorm_image(self.model_image[0,0].cpu().detach().numpy(), caption="Model Image"), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
            wb.log({"prediction": wandb_powernorm_image(prediction[0,0].cpu().detach().numpy(), caption="Prediction"), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
            wb.log({"sky image": wandb_powernorm_image(sky_image[0,0].cpu().detach().numpy(), caption="Sky Image"), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
            wb.log({"loss image": wandb_powernorm_image(loss_image[0,0].cpu().detach().numpy(), caption="Loss Image"), "major_cycle": self.current_cycle, "epoch": self.current_epoch})

            wb.log({
            f"loss/total_{train_cycle}": loss_dict['total'],
            f"loss/hierarchical_{train_cycle}": loss_dict['hierarchical'],
            f"loss/gradient_{train_cycle}": loss_dict['gradient'],
            f"loss/flux_{train_cycle}": loss_dict['flux'],
            f"loss/residual_{train_cycle}": loss_dict['residual'],
            f"stats/n_bright_pixels": loss_dict['n_bright_pixels'],
            f"stats/n_mid_pixels": loss_dict['n_mid_pixels'],
            f"stats/n_faint_pixels": loss_dict['n_faint_pixels'],
            "major_cycle": self.current_cycle,
            "epoch": self.current_epoch
            })

        else:
            self.unet.eval()
            with torch.no_grad():
                prediction = self.unet(residual_image, conditioning=train_cycle)
                self.model_image = torch.abs(self.model_image.detach() + prediction) if self.model_image is not None else prediction

    #@profile
    def major_cycle_step(self, x, y):
        print(f"Starting major cycle {self.current_cycle}, epoch {self.current_epoch} …")
        print("Loading vis")
        vis = x[0].cfloat()
        print("Loading uvw")
        uvw = x[1].float()
        print("Loading lmn")
        lmn = x[2].float()
        batchsize=int(vis.shape[0])
        y=y[:,0,:,:]+y[:,1,:,:]
        y = y.to(self.device)
        uvw = uvw.to(self.device)
        lmn = lmn.to(self.device)
        vis = vis.to(self.device)
        vis += self.generate_noise(vis.shape)
        #if self.current_cycle == 0:
        #    self.model_vis = torch.zeros(vis.shape, dtype=torch.cfloat, device=self.device)
        print("Data loaded, starting major cycle steps …")
        for i in range(self.current_cycle+1):
            if i ==0:
                self.model_vis = torch.zeros(vis.shape, dtype=torch.cfloat, device=self.device)
            residual_vis = vis.clone() - self.model_vis
            print(residual_vis.dtype)
            print(residual_vis.shape)
            print(vis.shape)
            print(uvw.shape)
            print(lmn.shape)
            residual_image = Parallel(n_jobs=batchsize, backend='threading', prefer='threads')(delayed(self.finufft.inufft)(residual_vis[k], lmn[k, 0, :], lmn[k, 1, :], lmn[k, 2, :], uvw[k, 0, :], uvw[k, 1, :], uvw[k, 2, :], return_torch=True) for k in range(batchsize))
            residual_image = torch.stack(residual_image, dim=0).to(self.device)#torch.tensor(residual_image).to(self.device)
            residual_image = residual_image.reshape((batchsize, self.pix_size, self.pix_size)).float()
            #residual_image = torch.abs(residual_image.reshape((batchsize, self.pix_size, self.pix_size))).float()
            if i == 0:
                self.dirty_image = residual_image.clone()
                self.model_image = torch.zeros_like(residual_image, device=self.device).float()
            #if i !=0:
                #residual_image.requires_grad_()
                #loss_residual=torch.mean(torch.abs(residual_image))/torch.max(torch.abs(self.dirty_image))
                #self.optimizer_major.zero_grad()
                #loss_residual.backward()
                #self.optimizer_major.step()
                #wb.log({f"Major Loop Loss {i}": loss_residual.item(), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
            print("Starting minor cycle step …")
            self.minor_cycle_step(residual_image, y, train_cycle=i)
            self.model_vis = Parallel(n_jobs=batchsize, backend='threading', prefer='threads')(delayed(self.finufft.nufft)(self.model_image.detach().clone()[k], lmn[k, 0, :], lmn[k, 1, :], lmn[k, 2, :], uvw[k, 0, :], uvw[k, 1, :], uvw[k, 2, :], return_torch=True) for k in range(batchsize))
            self.model_vis = torch.stack(self.model_vis, dim=0)#self.model_vis = torch.tensor(np.array(self.model_vis), device=self.device)
            self.resmean = torch.mean(torch.abs(torch.tensor(residual_vis)))
        return self.resmean
