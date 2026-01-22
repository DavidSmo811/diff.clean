import torch
import torch.nn as nn
import torch.nn.functional as F

    
class ConditionalLOFARScaler(nn.Module):
    def __init__(self, n_channels, num_conditions,
                 emb_dim=16, init_pos=1.0, init_neg=1.0, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.log_a_pos_base = nn.Parameter(
            torch.log(torch.ones(1, n_channels, 1, 1) * init_pos)
        )
        self.log_a_neg_base = nn.Parameter(
            torch.log(torch.ones(1, n_channels, 1, 1) * init_neg)
        )

        self.cond_emb = nn.Embedding(num_conditions, emb_dim)
        self.cond_proj = nn.Linear(emb_dim, 2 * n_channels)

        # 🔑 wichtig: Start bei delta = 0
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def _get_params(self, conditioning):
        emb = self.cond_emb(conditioning)
        delta = self.cond_proj(emb)   # [B, 2*C]

        delta_pos, delta_neg = delta.chunk(2, dim=1)

        log_a_pos = self.log_a_pos_base + delta_pos.unsqueeze(-1).unsqueeze(-1)
        log_a_neg = self.log_a_neg_base + delta_neg.unsqueeze(-1).unsqueeze(-1)
        log_a_pos = torch.clamp(log_a_pos, min=-10, max=10)
        log_a_neg = torch.clamp(log_a_neg, min=-10, max=10)

        return torch.exp(log_a_pos), torch.exp(log_a_neg)

    def forward(self, x, conditioning):
        a_pos, a_neg = self._get_params(conditioning)
        a_pos = a_pos.expand_as(x)
        a_neg = a_neg.expand_as(x)

        pos = x >= 0
        y = torch.empty_like(x)

        y[pos] = a_pos[pos] * torch.log1p(x[pos] / (a_pos[pos] + self.eps))
        y[~pos] = -a_neg[~pos] * torch.log1p(-x[~pos] / (a_neg[~pos] + self.eps))
        return y
    
    def inverse(self, y, conditioning):
        a_pos, a_neg = self._get_params(conditioning)
        a_pos = a_pos.expand_as(y)
        a_neg = a_neg.expand_as(y)

        pos = y >= 0
        z = torch.empty_like(y)
        z[pos] = a_pos[pos] * torch.expm1(y[pos] / (a_pos[pos] + self.eps))
        z[~pos] = -a_neg[~pos] * torch.expm1(-y[~pos] / (a_neg[~pos] + self.eps))

        return z

class LearnableLOFARScaler(nn.Module):
    def __init__(self, n_channels, init_pos=1.0, init_neg=1.0, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.log_a_pos = nn.Parameter(
            torch.log(torch.ones(1, n_channels, 1, 1) * init_pos)
        )
        self.log_a_neg = nn.Parameter(
            torch.log(torch.ones(1, n_channels, 1, 1) * init_neg)
        )

    def forward(self, x, no_conditioning):
        a_pos = torch.exp(self.log_a_pos)
        a_neg = torch.exp(self.log_a_neg)

        return torch.where(
            x >= 0,
            a_pos * torch.log1p(x / (a_pos + self.eps)),
            -a_neg * torch.log1p(-x / (a_neg + self.eps)),
        )

    def inverse(self, y, no_conditioning):
        a_pos = torch.exp(self.log_a_pos)
        a_neg = torch.exp(self.log_a_neg)

        return torch.where(
            y >= 0,
            a_pos * torch.expm1(y / (a_pos + self.eps)),
            -a_neg * torch.expm1(-y / (a_neg + self.eps)),
        )
    


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
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
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet mit einfachem FiLM (nur Beta), direkt aus Cycle-Index Embedding
    """
    def __init__(self, num_conditions, in_channels=1, out_channels=1, base_c=32, num_levels=4, bilinear=True, LOFAR_scaling=False, Conditional_Scaling=True):
        super().__init__()
        self.num_conditions = num_conditions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_c = base_c
        self.num_levels = num_levels
        self.bilinear = bilinear
        self.LOFAR_scaling = LOFAR_scaling
        self.Conditional_Scaling = Conditional_Scaling

        if self.LOFAR_scaling:
            self.input_scaler = LearnableLOFARScaler(
            n_channels=in_channels,
            init_pos=1.0,
            init_neg=1.0
            )
            self.output_scaler = LearnableLOFARScaler(
            n_channels=out_channels,
            init_pos=1.0,
            init_neg=1.0
            )
        if self.Conditional_Scaling:
            self.input_scaler = ConditionalLOFARScaler(
                n_channels=in_channels,
                num_conditions=num_conditions,
                init_pos=1.0,
                init_neg=1.0,
            )
            self.output_scaler = ConditionalLOFARScaler(
                n_channels=out_channels,
                num_conditions=num_conditions,
                init_pos=1,
                init_neg=1,
            )
        if self.Conditional_Scaling and self.LOFAR_scaling:
            raise ValueError("Either Conditional Scaling or Learnable Scaling can be used, not both.")
            

        enc_channels = [base_c * (2**i) for i in range(num_levels)]
        bottleneck_channels = base_c * (2**num_levels)

        self.inc = DoubleConv(in_channels, enc_channels[0])
        self.downs = nn.ModuleList([Down(enc_channels[i], enc_channels[i+1]) for i in range(num_levels-1)])
        self.down_bottleneck = Down(enc_channels[-1], bottleneck_channels)

        up_in_channels = [bottleneck_channels] + [enc_channels[i+1] for i in reversed(range(num_levels-1))]
        up_out_channels = list(reversed(enc_channels))
        self.ups = nn.ModuleList()
        self.ups.append(Up(bottleneck_channels + enc_channels[-1], enc_channels[-1], bilinear=bilinear))
        for i in range(num_levels-1):
            in_ch = enc_channels[num_levels-1 - i] + enc_channels[num_levels-2 - i]
            out_ch = enc_channels[num_levels-2 - i]
            self.ups.append(Up(in_ch, out_ch, bilinear=bilinear))

        self.outc = OutConv(enc_channels[0], out_channels)

        # FiLM Beta: kleine Embedding + lineare Projektion
        self.film_layout = enc_channels + [bottleneck_channels] + list(reversed(enc_channels))
        self.condition_embedding = nn.Embedding(num_conditions, 16)
        self.condition_proj = nn.Linear(16, sum(self.film_layout))

        # Output leicht positiv
        self.output_scale = nn.Parameter(torch.tensor(1e-2))#0.005
        self.output_bias = nn.Parameter(torch.tensor(0.001))#0.001

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_film(self, conditioning):
        if isinstance(conditioning, int):
            idx = torch.tensor([conditioning], dtype=torch.long, device=next(self.parameters()).device)
        else:
            idx = conditioning.long().to(next(self.parameters()).device)
        emb = self.condition_embedding(idx)
        beta_all = torch.sigmoid(self.condition_proj(emb))
        out = []
        cursor = 0
        for c in self.film_layout:
            beta = beta_all[:, cursor:cursor+c].unsqueeze(-1).unsqueeze(-1)
            cursor += c
            out.append(beta)
        return out

    def forward(self, x, conditioning=0):
        films = self._make_film(conditioning)
        film_idx = 0
        if self.LOFAR_scaling or self.Conditional_Scaling:
            if isinstance(conditioning, int):
                conditioning_in = torch.tensor([conditioning], dtype=torch.long, device=x.device)
            x = self.input_scaler(x, conditioning_in)
        x1 = self.inc(x)
        x1 = x1 + films[film_idx]
        film_idx += 1
        encs = [x1]
        xi = x1
        for down in self.downs:
            xi = down(xi)
            xi = xi + films[film_idx]
            film_idx += 1
            encs.append(xi)

        xb = self.down_bottleneck(encs[-1])
        xb = xb + films[film_idx]
        film_idx += 1

        x_up = xb
        for i, up in enumerate(self.ups):
            skip = encs[-1 - i]
            x_up = up(x_up, skip)
            if i < len(self.ups) - 1:
                x_up = x_up + films[film_idx]
                film_idx += 1

        out = self.outc(x_up)
        out = out * self.output_scale + self.output_bias
        if self.LOFAR_scaling or self.Conditional_Scaling:
            if isinstance(conditioning, int):
                conditioning_out = torch.tensor([conditioning], dtype=torch.long, device=x.device)
            out = self.output_scaler.inverse(out, conditioning_out)
        return out