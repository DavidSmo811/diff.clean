import torch
import torch.nn.functional as F
import numpy as np
import cupy as cp
import os
import wandb as wb
from joblib import Parallel, delayed
from finufft import CupyFinufft


class reconstruction_loop:
    def __init__(self, data_loader, unet, optimizer,num_max_major_cycle, epochs, pix_size=512, fov_arcsec=6000, eps=1e-6, max_resmean=1e-3, device=None):
        self.dataloader = data_loader
        self.unet = unet
        self.optimizer = optimizer
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

        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.unet.to(self.device)

    def save_checkpoint(self, name):
        path = os.path.join(self.checkpoint_dir, name)
        torch.save(self.unet.state_dict(), path)
        print(f"[Checkpoint] Saved: {path}")

    def __iter__(self):
        self.current_cycle = 0
        self.current_epoch = 0
        self.model_vis = None
        self.model_image = None
        return self
    
    def __next__(self):
        if self.current_cycle > self.num_max_major_cycle:
            print("Reached maximum number of major cycles.")
            self.save_checkpoint(f"unet_final_cycle_{self.current_cycle}.pt")
            raise StopIteration
        if self.resmean <= self.max_resmean and self.resmean != -1:
            print(f"Stopping at major cycle {self.current_cycle} with residual mean {self.resmean}")
            self.save_checkpoint(f"unet_converged_cycle_{self.current_cycle}.pt")
            raise StopIteration
        self.current_epoch +=1
        self.model_image = None
        if self.current_epoch > self.epochs:
            self.current_cycle += 1
            self.current_epoch = 0
            self.save_checkpoint(f"unet_cycle_{self.current_cycle}.pt")
        return self.major_cycle_step(self.dataloader)
    

    def loss_percentile_scheduler(self, ground_truth, start_percentile=99.99, end_percentile=70.0, k=5.0):
        x = torch.linspace(0, 1, self.num_max_major_cycle, dtype=ground_truth.dtype).to(self.device)  # linear 0..1
        k = 5.0  # >1 → starkes Anhäufen am Ende
        y = start_percentile + (end_percentile - start_percentile) * (1 - (1 - x)**k)
        y=torch.flip(y, dims=[0])
        print(ground_truth.dtype)
        print(y[self.current_cycle].dtype)
        percentile_gt = torch.quantile(torch.abs(ground_truth), y[self.current_cycle] / 100.0)
        filter_image = torch.where(torch.abs(ground_truth) >= percentile_gt, ground_truth, torch.zeros_like(ground_truth))
        return filter_image

    

    def minor_cycle_step(self, residual_image, sky_image, train_cycle, logging_step=100):
        print(self.current_cycle, self.current_epoch)
        if residual_image.dim() == 3:
            residual_image = residual_image.unsqueeze(1)  # (B,1,H,W)

        if sky_image.dim() == 3:
            sky_image = sky_image.unsqueeze(1)

        train = train_cycle==self.current_cycle
        if train:
            loss_image=self.loss_percentile_scheduler(sky_image)
            self.unet.train()
            self.optimizer.zero_grad()
            prediction = self.unet(residual_image, conditioning=train_cycle)
            self.model_image = self.model_image + prediction if self.model_image is not None else prediction
            loss = F.l1_loss(self.model_image, loss_image)
            loss.backward()
            self.optimizer.step()
            if self.current_epoch % logging_step == 0:
                wb.log({"minor_cycle_loss": loss.item(), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
                wb.log({"residual image": wb.Image(residual_image[0,0].cpu().detach().numpy(), caption="Residual Image"), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
                wb.log({"model image": wb.Image(self.model_image[0,0].cpu().detach().numpy(), caption="Model Image"), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
                wb.log({"prediction": wb.Image(prediction[0,0].cpu().detach().numpy(), caption="Prediction"), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
                wb.log({"sky image": wb.Image(sky_image[0,0].cpu().detach().numpy(), caption="Sky Image"), "major_cycle": self.current_cycle, "epoch": self.current_epoch})
                wb.log({"loss image": wb.Image(loss_image[0,0].cpu().detach().numpy(), caption="Loss Image"), "major_cycle": self.current_cycle, "epoch": self.current_epoch})

        else:
            self.unet.eval()
            with torch.no_grad():
                prediction = self.unet(residual_image, conditioning=train_cycle)
                self.model_image = self.model_image + prediction if self.model_image is not None else prediction

    def major_cycle_step(self, data_in):
        print(f"Starting major cycle {self.current_cycle}, epoch {self.current_epoch} …")
        for x,y in data_in:
            print("Loading vis")
            vis = x[0].cfloat()
            print("Loading uvw")
            uvw = x[1].float()
            print("Loading lmn")
            lmn = x[2].float()
            batchsize=int(vis.shape[0])
            y = y.to(self.device)
            if self.current_cycle == 0:
                self.model_vis = torch.zeros(vis.shape, dtype=torch.cfloat)
            print("Data loaded, starting major cycle steps …")
            for i in range(self.current_cycle+1):
                residual_vis = vis - self.model_vis
                print(residual_vis.dtype)
                print(residual_vis.shape)
                print(vis.shape)
                print(uvw.shape)
                print(lmn.shape)
                residual_image = Parallel(n_jobs=batchsize, backend='threading', prefer='threads')(delayed(self.finufft.inufft)(residual_vis[k], lmn[k, 0, :], lmn[k, 1, :], lmn[k, 2, :], uvw[k, 0, :], uvw[k, 1, :], uvw[k, 2, :]) for k in range(batchsize))
                residual_image = torch.tensor(residual_image).to(self.device)
                residual_image = torch.abs(residual_image.reshape((batchsize, self.pix_size, self.pix_size))).float()
                print("Starting minor cycle step …")
                self.minor_cycle_step(residual_image, y, train_cycle=i)
                self.model_vis = Parallel(n_jobs=batchsize, backend='threading', prefer='threads')(delayed(self.finufft.nufft)(self.model_image, lmn[k, 0, :], lmn[k, 1, :], lmn[k, 2, :], uvw[k, 0, :], uvw[k, 1, :], uvw[k, 2, :], return_torch=True) for k in range(batchsize))
                self.resmean = torch.mean(torch.abs(torch.tensor(residual_vis)))
