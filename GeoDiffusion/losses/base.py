import torch
import torch.nn as nn


class KLSurfaceField(nn.Module):
    def __init__(self, kl_weight=0.00001):
        super().__init__()

        self.kl_weight = kl_weight
    
    def forward(self, posteriors,
                recon_feats,
                labels):
        recon_loss = nn.MSELoss()(recon_feats, labels)
        if posteriors is not None:
            kl_loss = posteriors.kl(dims=(1,2))
            kl_loss = torch.mean(kl_loss)
        else:
            kl_loss = torch.tensor(0.0, device=recon_feats.device)
        
        loss = recon_loss + self.kl_weight * kl_loss
        loss_log = {"recon_loss": recon_loss.item(), "kl_loss": kl_loss.item()}
        return loss, loss_log