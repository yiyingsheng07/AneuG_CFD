# %%
import torch
from torch.utils.data import DataLoader
from datasets.wss import read_ansys_csv
import os
from models.field_vae.base import SurfaceFieldAutoEncoder
import torch.nn as nn

# %%
from datasets.wss import WSSPeakDataset
from losses.base import KLSurfaceField

batch_size = 32
train_split = 0.9
encode_size = 16800
decode_size = 7200

root_dir = '/media/yaplab/HDD_Storage/wenhao/datasets/AneuG_CFD/peak_wss'
dataset = WSSPeakDataset(root_dir, encode_size=encode_size, decode_size=decode_size)
train_size = int(train_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
loss_module = KLSurfaceField(kl_weight=0.00001)
recon_loss_module = nn.MSELoss()


# %%
device = torch.device("cuda:0")
num_latents = 128
feature_dim = 1
embed_dim = 16
num_freqs = 8
width = 768 // 4
heads = 12 // 4
num_encoder_layers = 4
num_decoder_layers = 8

SurfaceFieldVAE = SurfaceFieldAutoEncoder(device=device,
                                          num_latents=num_latents,
                                          feature_dim=feature_dim,
                                          embed_dim=embed_dim,
                                          num_freqs=num_freqs,
                                          width=width,
                                          heads=heads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)
# SurfaceFieldVAE = nn.DataParallel(SurfaceFieldVAE, device_ids=[0, 1])
SurfaceFieldVAE.to(device)

# %%
import wandb
log_wandb = True
meta = "128_16_4_8"
if log_wandb:
    wandb.login()
    run = wandb.init(project="geodiffusion",
                     name=meta)
    
optimizer = torch.optim.AdamW(SurfaceFieldVAE.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.5)

# %%
for epoch in range(100000+1):
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = {key: value.to(device) for key, value in data.items()}
        coords, feats, recon_coords, recon_feats_true = data['coords'], data['feats'], data['recon_coords'], data['recon_feats']
        recon_feats_true = recon_feats_true.squeeze(-1)
        recon_feats_pred, center_pos, kl_loss = SurfaceFieldVAE(coords, feats, recon_coords, sample_posterior=True)
        kl_loss = torch.mean(kl_loss)
        recon_loss = recon_loss_module(recon_feats_pred, recon_feats_true)
        loss = recon_loss + kl_loss * 0.00001
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        recon_loss_test = 0.0
        for j, data in enumerate(test_loader):
            data = {key: value.to(device) for key, value in data.items()}
            coords, feats, recon_coords, recon_feats_true = data['coords'], data['feats'], data['recon_coords'], data['recon_feats']
            recon_feats_true = recon_feats_true.squeeze(-1)
            recon_feats_pred, center_pos, _ = SurfaceFieldVAE(coords, feats, recon_coords, sample_posterior=True)
            loss_test = recon_loss_module(recon_feats_pred, recon_feats_true)
            recon_loss_test += loss_test.item() / len(test_loader)
        print(f'Epoch: {epoch}, Test Loss: {recon_loss_test}')
    
    if epoch % 1000 == 0:
        save_path = os.path.join('./checkpoints', meta,  f'epoch_{epoch}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(SurfaceFieldVAE.state_dict(), save_path)
    
    log_dict = {'recon_loss': recon_loss.item(), 'kl_loss': kl_loss.item(), 'test_loss': recon_loss_test, 'lr': optimizer.param_groups[0]['lr']}
    print(log_dict)
    if log_wandb:
        wandb.log(log_dict, step=epoch)
    
    scheduler.step()
wandb.finish()


"""
python first_stage_wss.py

chmod -R 777 /media/yaplab/HDD_Storage/wenhao/AneuG_CFD

"""