import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveRadioReconstructionLoss(nn.Module):
    """
    Adaptiver Loss der ERST Punktquellen rekonstruiert, DANN diffuse Emission.
    Strategie: Dynamische Anpassung basierend auf Residual-Qualität.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
    def compute_hierarchical_percentile_loss(self, pred, target, dirty, 
                                             bright_perc=99.95, mid_perc=97.0, faint_perc=90.0):
        """
        Hierarchischer Loss: Verschiedene Perzentile mit UNTERSCHIEDLICHER Gewichtung.
        Helle Quellen bekommen VIEL mehr Gewicht damit sie zuerst gut werden.
        """
        total_loss = 0.0
        
        # 1. SEHR HELLE PUNKTQUELLEN (99.95%) - HÖCHSTE PRIORITÄT
        threshold_bright = torch.quantile(torch.abs(target), bright_perc / 100.0)
        mask_bright = (torch.abs(target) >= threshold_bright).float()
        
        if mask_bright.sum() > 0:
            # Nutze L1 für Punktquellen (präziser für Peaks)
            loss_bright = nn.L1Loss()(pred * mask_bright, target * mask_bright) / (
                nn.L1Loss()(dirty * mask_bright, target * mask_bright) + 1e-8
            )
            # SEHR HOHES GEWICHT für Punktquellen
            total_loss += 10.0 * loss_bright  
        
        # 2. MITTLERE STRUKTUREN (97%) - nur wenn helle Quellen schon gut sind
        threshold_mid = torch.quantile(torch.abs(target), mid_perc / 100.0)
        mask_mid = ((torch.abs(target) >= threshold_mid) & (torch.abs(target) < threshold_bright)).float()
        
        if mask_mid.sum() > 0:
            loss_mid = nn.L1Loss()(pred * mask_mid, target * mask_mid) / (
                nn.L1Loss()(dirty * mask_mid, target * mask_mid) + 1e-8
            )
            # Mittleres Gewicht
            total_loss += 3.0 * loss_mid
        
        # 3. DIFFUSE EMISSION (90%) - niedrigste Priorität
        mask_faint = (torch.abs(target) < threshold_mid).float()
        
        if mask_faint.sum() > 0:
            # Für diffuse Emission: Log-Loss ist besser (Dynamikbereich)
            loss_faint = nn.HuberLoss()(
                torch.log(torch.abs(pred * mask_faint) + 1e-10),#1e-7 vorher
                torch.log(torch.abs(target * mask_faint) + 1e-10)#1e-7 vorher
            )
            # Kleinstes Gewicht
            total_loss += 1.0 * loss_faint
        
        return total_loss, mask_bright, mask_mid, mask_faint
    
    def compute_adaptive_gradient_loss(self, pred, target, bright_mask):
        """
        Gradient loss der NUR um Punktquellen herum aktiv ist.
        Bestraft Ring-Artefakte um die hellen Quellen.
        """
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=pred.dtype, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=pred.dtype, device=self.device).view(1, 1, 3, 3)
        
        grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
        grad_target_x = F.conv2d(target, sobel_x, padding=1)
        grad_target_y = F.conv2d(target, sobel_y, padding=1)
        
        grad_pred = torch.sqrt(grad_pred_x**2 + grad_pred_y**2 + 1e-8)
        grad_target = torch.sqrt(grad_target_x**2 + grad_target_y**2 + 1e-8)
        
        # Erweitere die Maske um Punktquellen (damit auch der Ring erwischt wird)
        kernel_size = 15  # Ring-Radius in Pixeln
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size**2)
        extended_mask = F.conv2d(bright_mask, kernel, padding=kernel_size//2)
        extended_mask = (extended_mask > 0.01).float()
        
        # Gradient Loss nur in dieser erweiterten Region
        grad_loss = nn.L1Loss()(grad_pred * extended_mask, grad_target * extended_mask)
        
        return grad_loss
    
    def compute_residual_statistics_loss(self, pred, target):
        """
        Loss basierend auf Residual-Statistiken.
        Wenn Punktquellen gut sind, sollte das Residual nur noch Rauschen sein.
        """
        # Simuliere was das Residual wäre
        simulated_residual = target - pred
        
        # Peak im Residual
        residual_peak = torch.max(torch.abs(simulated_residual), dim=-1, keepdim=True)[0]
        residual_peak = torch.max(residual_peak, dim=-2, keepdim=True)[0]
        target_peak = torch.max(torch.abs(target), dim=-1, keepdim=True)[0]
        target_peak = torch.max(target_peak, dim=-2, keepdim=True)[0]
        
        # Residual-Peak sollte klein sein relativ zum Original-Peak
        peak_ratio = residual_peak / (target_peak + 1e-8)
        loss_peak = torch.mean(peak_ratio**2)
        
        return loss_peak
    
    def compute_flux_conservation_loss(self, pred, target, bright_mask):
        """
        Flux Conservation mit FOKUS auf helle Quellen.
        """
        # Total Flux für die ganze Region
        pred_sum_total = torch.sum(pred, dim=[2, 3])
        target_sum_total = torch.sum(target, dim=[2, 3])
        loss_total = nn.L1Loss()(pred_sum_total, target_sum_total) / (torch.abs(target_sum_total) + 1e-8).mean()
        
        # Flux nur für helle Quellen (wichtiger!)
        pred_sum_bright = torch.sum(pred * bright_mask, dim=[2, 3])
        target_sum_bright = torch.sum(target * bright_mask, dim=[2, 3])
        loss_bright = nn.L1Loss()(pred_sum_bright, target_sum_bright) / (torch.abs(target_sum_bright) + 1e-8).mean()
        
        return 0.3 * loss_total + 0.7 * loss_bright
    
    # def compute_negative_penalty(self, pred, bright_mask):
    #     """
    #     Bestraft negative Werte, besonders bei hellen Quellen.
    #     """
    #     neg_all = torch.mean(F.relu(-pred)**2)
    #     neg_bright = torch.mean(F.relu(-pred * bright_mask)**2)
        
    #     return 0.3 * neg_all + 0.7 * neg_bright
    
    def compute_cycle_adaptive_weights(self, epoch, total_epochs):
        """
        KRITISCH: Gewichte ändern sich innerhalb eines Zyklus!
        
        Strategie:
        - Früh im Zyklus (Epoche 1-3): Fokus auf Punktquellen
        - Mitte (Epoche 4-6): Balance
        - Spät (Epoche 7+): Mehr Diffuse + Ring-Kontrolle
        """
        epoch_progress = epoch / max(total_epochs, 1)
        
        # Frühe Epochen: Punktquellen dominieren
        if epoch_progress < 0.4:
            return {
                'hierarchical': 0.55,   # Hoher Fokus auf Punktquellen
                'gradient': 0.1,      # Wenig Gradient-Kontrolle
                'flux': 0.2,           # Flux wichtig
                'residual': 0.15,      # Residual-Peaks eliminieren
                #'negative': 0.1
            }
        # Mittlere Epochen: Balance
        elif epoch_progress < 0.7:
            return {
                'hierarchical': 0.45,
                'gradient': 0.2,      # Mehr Gradient
                'flux': 0.2,
                'residual': 0.15,
                #'negative': 0.1
            }
        # Späte Epochen: Diffuse + Artefakt-Kontrolle
        else:
            return {
                'hierarchical': 0.375,   # Immer noch wichtig
                'gradient': 0.325,      # VIEL Gradient (Ringe!)
                'flux': 0.2,
                'residual': 0.1,
                #'negative': 0.15       # Mehr Negative-Kontrolle
            }
    
    def forward(self, pred, target, dirty, epoch, total_epochs):
        """
        Hauptfunktion: Kombiniert alle Losses adaptiv.
        """
        # Hole adaptive Gewichte
        weights = self.compute_cycle_adaptive_weights(epoch, total_epochs)
        
        # 1. Hierarchischer Percentile Loss (hauptsächlich)
        loss_hier, mask_bright, mask_mid, mask_faint = self.compute_hierarchical_percentile_loss(
            pred, target, dirty
        )
        
        # 2. Gradient Loss (Ring-Artefakte)
        loss_grad = self.compute_adaptive_gradient_loss(pred, target, mask_bright)
        
        # 3. Flux Conservation
        loss_flux = self.compute_flux_conservation_loss(pred, target, mask_bright)
        
        # 4. Residual Statistics
        loss_residual = self.compute_residual_statistics_loss(pred, target)

        
        # 5. Negative Penalty
        #loss_neg = self.compute_negative_penalty(pred, bright_mask)
        
        # Kombiniere
        total_loss = (
            weights['hierarchical'] * loss_hier +
            weights['gradient'] * loss_grad +
            weights['flux'] * loss_flux +
            weights['residual'] * loss_residual
            #+weights['negative'] * loss_neg  
        )   
        
        # Statistiken für Monitoring
        loss_dict = {
            'total': total_loss,
            'hierarchical': loss_hier,
            'gradient': loss_grad,
            'flux': loss_flux,
            'residual': loss_residual,
            #'negative': loss_neg.item(),
            'n_bright_pixels': mask_bright.sum(),
            'n_mid_pixels': mask_mid.sum(),
            'n_faint_pixels': mask_faint.sum(),
        }
        
        return total_loss, loss_dict
