import torch
import numpy as np
from joblib import Parallel, delayed
from finufft import CupyFinufft


class reconstruction_loop:
    def __init__(self, data_loader, unet, optimizer,num_max_major_cycle, epochs, pix_size=512, fov_arcsec=6000, eps=1e-6, max_resmean=1e-3):
        self.dataloader = data_loader
        self.unet = unet
        self.optimizer = optimizer
        self.finufft = CupyFinufft(image_size=pix_size, fov_arcsec=fov_arcsec, eps=eps)
        self.num_max_major_cycle = num_max_major_cycle
        self.model_image = None
        self.model_vis = None
        self.max_resmean = max_resmean
        self.epochs = epochs
        self.resmean = -1
        self.current_epoch = 0
        self.current_cycle = 0


    def __iter__(self):
        self.current_cycle = 0
        self.current_epoch = 0
        self.model_vis = None
        self.model_image = None
        return self
    
    def __next__(self):
        if self.current_cycle > self.num_max_major_cycle:
            raise StopIteration
        if self.resmean <= self.max_resmean and self.resmean != -1:
            raise StopIteration
        self.current_epoch +=1
        if self.current_epoch > self.epochs:
            self.current_cycle += 1
            self.current_epoch = 0
        return self.major_cycle_step(self.dataloader)
    

    def loss_percentile_scheduler(self, ground_truth, start_percentile=99.99, end_percentile=70.0, k=5.0):
        x = torch.linspace(0, 1, self.num_max_major_cycle)  # linear 0..1
        k = 5.0  # >1 → starkes Anhäufen am Ende
        y = start_percentile + (end_percentile - start_percentile) * (1 - (1 - x)**k)
        y=torch.flip(y, dims=[0])
        percentile_gt = torch.quantile(torch.abs(ground_truth), y[self.current_cycle] / 100.0)

    

    def minor_cycle_step(self, residual_image, sky_image, train_cycle):
        train = train_cycle==self.current_cycle
        if train:
            loss_image=self.loss_percentile_scheduler(sky_image)
            self.unet.train()
            self.optimizer.zero_grad()
        


    def major_cycle_step(self, data_in):
        for x,y in data_in:
            vis = x[0]
            uvw = x[1]
            lmn = x[2]
        train=False
        batchsize=int(vis.shape[0])
        if self.current_cycle == 0:
            self.model_vis = torch.zeros(vis.shape, dtype=torch.cfloat)
        for i in range(self.current_cycle+1):
            if i == self.current_cycle:
                train=True
            residual_vis = vis - self.model_vis
            residual_image = Parallel(n_jobs=batchsize, backend='threading', prefer='threads')(delayed(self.finufft.ift)(residual_vis, lmn[i,:, 0], lmn[i,:, 1], lmn[i,:, 2], uvw[i,:, 0], uvw[i,:, 1], uvw[i,:, 2]) for i in range(batchsize))
            residual_image = torch.tensor(residual_image)
            prediction_image = self.minor_cycle_step(residual_image, y, train_cycle=i)
            self.model_image = self.model_image + prediction_image if self.model_image is not None else prediction_image
            self.model_vis = Parallel(n_jobs=batchsize, backend='threading', prefer='threads')(delayed(self.finufft.ft)(self.model_image, lmn[i,:, 0], lmn[i,:, 1], lmn[i,:, 2], uvw[i,:, 0], uvw[i,:, 1], uvw[i,:, 2]) for i in range(batchsize))
            self.resmean = torch.mean(torch.abs(torch.tensor(residual_vis)))
