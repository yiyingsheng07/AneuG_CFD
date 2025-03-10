from models.field_vae.base import SurfaceFieldAutoEncoder
import torch


device = torch.device("cuda:0")
num_latents = 128
feature_dim = 3
embed_dim = 128
num_freqs = 8
width = 256
heads = 12
num_encoder_layers = 6
num_decoder_layers = 12

SurfaceFieldVAE = SurfaceFieldAutoEncoder(device=device,
                                          num_latents=num_latents,
                                          feature_dim=feature_dim,
                                          embed_dim=embed_dim,
                                          num_freqs=num_freqs,
                                          width=width,
                                          heads=heads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)