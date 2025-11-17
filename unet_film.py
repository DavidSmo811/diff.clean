"""
Unet.py

UNet with FiLM-style conditioning on the number of major cycles (an integer).

Usage example:
    from Unet import UNet
    unet = UNet(num_conditions=num_max_major_cycle+1, in_channels=1, out_channels=1)
    out = unet(x, conditioning=cycle_idx)   # cycle_idx can be int or tensor (B,) of ints

The conditioning is embedded and produces per-level scale (gamma) and shift (beta)
that are applied after each encoder/decoder block (FiLM):  x = gamma * x + beta

This file is intentionally standalone and only depends on PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Downscaling with maxpool then DoubleConv"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upscaling then DoubleConv. Uses bilinear upsampling by default."""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            # transposed conv
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1 is the upsampled decoder feature, x2 is the skip connection from encoder
        x1 = self.up(x1)
        # pad if necessary (when input sizes are odd)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet with FiLM conditioning.

    Parameters
    ----------
    num_conditions : int
        Number of distinct conditioning indices (e.g. num_max_major_cycle + 1).
    in_channels, out_channels : int
    base_c : int
        Base channel number; channels double each level.
    num_levels : int
        Number of down/up levels (default 4 produces 5 resolution levels including bottleneck).
    bilinear : bool
        Whether to use bilinear upsampling (True) or transposed conv (False).
    """
    def __init__(self, num_conditions, in_channels=1, out_channels=1, base_c=32, num_levels=4, bilinear=True):
        super().__init__()
        assert num_conditions >= 1, "num_conditions must be >= 1"

        self.num_conditions = num_conditions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_c = base_c
        self.num_levels = num_levels
        self.bilinear = bilinear

        # Build encoder and decoder channel sizes
        enc_channels = [base_c * (2 ** i) for i in range(num_levels)]
        bottleneck_channels = base_c * (2 ** num_levels)

        # first conv
        self.inc = DoubleConv(in_channels, enc_channels[0])
        # down blocks
        self.downs = nn.ModuleList([Down(enc_channels[i], enc_channels[i+1]) for i in range(num_levels-1)])
        # final down to bottleneck
        self.down_bottleneck = Down(enc_channels[-1], bottleneck_channels)

        # up blocks (note concatenation doubles channels)
        # up from bottleneck -> enc_channels[-1]
        up_in_channels = [bottleneck_channels,] + [enc_channels[i+1] for i in reversed(range(num_levels-1))]
        up_out_channels = list(reversed(enc_channels))
        self.ups = nn.ModuleList()
        # first up: input channels = bottleneck + enc_channels[-1]
        self.ups.append(Up(bottleneck_channels + enc_channels[-1], enc_channels[-1], bilinear=bilinear))
        # remaining ups
        for i in range(num_levels-1):
            in_ch = enc_channels[num_levels-1 - i] + enc_channels[num_levels-2 - i]
            out_ch = enc_channels[num_levels-2 - i]
            self.ups.append(Up(in_ch, out_ch, bilinear=bilinear))

        self.outc = OutConv(enc_channels[0], out_channels)

        # Conditioning MLP: map conditioning index -> FiLM parameters (gamma,beta) for each level
        # We'll create one pair (gamma,beta) per feature-channel in each level (including bottleneck and decoder outputs)
        # Collect sizes per location where we'll apply FiLM (after inc, after each down conv, after bottleneck, after each up conv)
        film_channels = [enc_channels[0]] + enc_channels[1:] + [bottleneck_channels] + list(reversed(enc_channels))
        # We'll apply FiLM at these positions in the forward pass. Total parameters per condition = sum(2 * c for c in film_channels)
        total_film_params = sum([2 * c for c in film_channels])

        # Embedding + small MLP to produce film parameters
        # Use embedding for discrete conditions (indices). If you prefer continuous conditioning, pass floats and the module will convert.
        self.condition_embedding = nn.Embedding(num_conditions, min(64, num_conditions))
        # MLP size
        emb_dim = self.condition_embedding.embedding_dim
        hidden = max(emb_dim * 2, 128)
        self.film_mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, total_film_params)
        )

        # store film channel layout for easy slicing
        self.film_layout = film_channels

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_film(self, conditioning):
        """
        conditioning: int, or tensor of shape (B,) with dtype long or int.
        returns lists of (gamma,beta) tensors for each film location, each shaped (B, C, 1, 1)
        """
        if isinstance(conditioning, int):
            idx = torch.tensor([conditioning], dtype=torch.long, device=next(self.parameters()).device)
        elif isinstance(conditioning, torch.Tensor):
            # Accept scalar tensor, or 1D tensor of indices
            if conditioning.dim() == 0:
                idx = conditioning.unsqueeze(0).long().to(next(self.parameters()).device)
            else:
                idx = conditioning.long().to(next(self.parameters()).device)
        else:
            # try convert
            idx = torch.tensor(conditioning, dtype=torch.long, device=next(self.parameters()).device)

        emb = self.condition_embedding(idx)  # (B, emb_dim)
        film_params = self.film_mlp(emb)     # (B, total_film_params)

        # split into gamma and beta per layout
        out = []
        cursor = 0
        B = film_params.shape[0]
        for c in self.film_layout:
            nparams = 2 * c
            chunk = film_params[:, cursor:cursor + nparams]  # (B, 2*c)
            cursor += nparams
            gamma = chunk[:, :c].unsqueeze(-1).unsqueeze(-1)  # (B, c, 1, 1)
            beta = chunk[:, c:].unsqueeze(-1).unsqueeze(-1)
            out.append((gamma, beta))
        return out

    def forward(self, x, conditioning=0):
        """
        x: (B, C, H, W)
        conditioning: int or tensor (B,) of ints
        """
        # Build FiLM parameters
        films = self._make_film(conditioning)
        film_idx = 0

        # encoder
        x1 = self.inc(x)
        # apply FiLM after inc
        gamma, beta = films[film_idx]
        film_idx += 1
        x1 = x1 * (1.0 + gamma) + beta

        encs = [x1]
        xi = x1
        for down in self.downs:
            xi = down(xi)
            gamma, beta = films[film_idx]
            film_idx += 1
            xi = xi * (1.0 + gamma) + beta
            encs.append(xi)

        # bottleneck
        xb = self.down_bottleneck(encs[-1])
        gamma, beta = films[film_idx]
        film_idx += 1
        xb = xb * (1.0 + gamma) + beta

        # decoder
        x_up = xb
        # first up uses last encoder's skip
        for i, up in enumerate(self.ups):
            # choose corresponding skip: encs in reverse order
            skip = encs[-1 - i]
            x_up = up(x_up, skip)
            gamma, beta = films[film_idx]
            film_idx += 1
            x_up = x_up * (1.0 + gamma) + beta

        out = self.outc(x_up)
        return out


if __name__ == "__main__":
    # quick smoke test
    model = UNet(num_conditions=6, in_channels=1, out_channels=1, base_c=16, num_levels=3)
    x = torch.randn(2, 1, 512, 512)
    y = model(x, conditioning=torch.tensor([0, 3]))
    print(y[0,0])
    print("Output shape:", y.shape)
