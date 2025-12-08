import torch
import torch.nn as nn
from joblib import Parallel, delayed
import wandb as wb
from PIL import Image
from finufft_dlpack import CupyFinufft
import io

class InferenceLoop:
    """
    Inference-only loop that mirrors die Major/Minor-Zyklen deines Trainingsloops,
    aber ohne backprop / optimizer.
    """
    def __init__(self, unet,
                 pix_size=512,
                 fov_arcsec=6000,
                 eps=1e-6,
                 device=None,
                 use_wandb=False):
        self.unet = unet
        self.finufft = CupyFinufft(image_size=pix_size, fov_arcsec=fov_arcsec, eps=eps)
        self.pix_size = pix_size
        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.unet.to(self.device)
        self.model_image = None      # (B,1,H,W) float tensor
        self.model_vis = None        # (B, ... ) complex tensor
        self.resmean = None
        self.use_wandb = use_wandb

    def load_checkpoint(self, ckpt_path, map_location=None):
        """Lädt state_dict eines trainierten UNet."""
        map_location = map_location if map_location is not None else self.device
        sd = torch.load(ckpt_path, map_location=map_location)
        self.unet.load_state_dict(sd)
        self.unet.eval()
        print(f"[Inference] Loaded checkpoint: {ckpt_path}")

    def _minor_cycle_infer(self, residual_image, dirty_image, train_cycle):
        """
        Inference-Version des Minor Cycle:
        - nutzt exakt dieselben Schritte wie das Training
        - 2-Kanal Input: [residual_image, previous_model_image]
        - Normierung: / max(|dirty_image|)
        - prediction = abs(prediction)
        - model_image = abs(model_image + prediction)
        """
        if residual_image.dim() == 3:
            residual_image = residual_image.unsqueeze(1)  # (B,1,H,W)

        if self.model_image is None:
            # analog Training: model_image vor MinorCycle = 0
            self.model_image = torch.zeros_like(residual_image, device=self.device)

        # dirty_image wird genauso wie Training behandelt
        if dirty_image.dim() == 3:
            dirty_image = dirty_image.unsqueeze(1)

        max_dirty = torch.max(torch.abs(dirty_image)).detach()
        if max_dirty == 0:
            max_dirty = torch.tensor(1.0, device=self.device)

        # INPUT wie im Training: residual + prev model
        input_image = torch.cat(
            [residual_image.detach(), self.model_image.detach()], dim=1
        )
        input_image = input_image / max_dirty

        with torch.no_grad():
            self.unet.eval()
            prediction = self.unet(input_image, conditioning=train_cycle)
            prediction = prediction * max_dirty
            prediction = torch.abs(prediction)  # wie Training

            # Das selbe Addieren wie Training:
            self.model_image = prediction #torch.abs(self.model_image.detach() + prediction)
            #Hier anpassen, nach dem nächsten Training, dass self.model_image = prediction genommen wird --> Aber erst beim Run ...prediction_is_modelimage

        return

    def run_major_minor(self, vis, uvw, lmn, num_major_cycles=3):
        vis = vis.to(self.device)
        uvw = uvw.to(self.device)
        lmn = lmn.to(self.device)
        batchsize = int(vis.shape[0])

        self.model_image = None
        self.model_vis = torch.zeros_like(vis, dtype=vis.dtype, device=self.device)

        dirty_image = None
        res_image=[]

        for major_i in range(num_major_cycles):
            residual_vis = vis.clone() - self.model_vis

            # iNUFFT
            residual_image_list = Parallel(n_jobs=batchsize)(
                delayed(self.finufft.inufft)(
                    residual_vis[k],
                    lmn[k,0], lmn[k,1], lmn[k,2],
                    uvw[k,0], uvw[k,1], uvw[k,2],
                    return_torch=True
                ) for k in range(batchsize)
            )
            residual_image = torch.stack(residual_image_list, dim=0).to(self.device)
            residual_image = residual_image.reshape(batchsize, self.pix_size, self.pix_size).float()
            res_image.append(residual_image.cpu().numpy())

            # Dirty Image setzen wie im Training
            if major_i == 0:
                dirty_image = residual_image.clone()

            # minor cycle
            self._minor_cycle_infer(residual_image, dirty_image, train_cycle=major_i)

            # Forward NUFFT
            model_vis_list = Parallel(n_jobs=batchsize)(
                delayed(self.finufft.nufft)(
                    self.model_image.detach().clone()[k],
                    lmn[k,0], lmn[k,1], lmn[k,2],
                    uvw[k,0], uvw[k,1], uvw[k,2],
                    return_torch=True
                ) for k in range(batchsize)
            )
            self.model_vis = torch.stack(model_vis_list, dim=0)

            self.resmean = torch.mean(torch.abs(residual_vis))

            print(f"[Inference] Major {major_i}/{num_major_cycles} — resmean {self.resmean.item():.6f}")

        return {
            "model_image": self.model_image,
            "model_vis": self.model_vis,
            "res_image": res_image,
            "resmean": self.resmean,
        }