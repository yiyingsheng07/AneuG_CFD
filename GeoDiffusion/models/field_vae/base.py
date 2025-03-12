# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Optional
from einops import repeat
import math
import sys
from typing import Tuple, List, Optional
sys.path.append(".../")

from models.modules.checkpoint import checkpoint
from models.modules.embedder import FourierEmbedder
from models.modules.distributions import DiagonalGaussianDistribution
from models.modules.transformer_blocks import (
    ResidualCrossAttentionBlock,
    Transformer
)


class ShapeAsLatentModule(nn.Module):
    latent_shape: Tuple[int, int]

    def __init__(self, *args, **kwargs):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def query_geometry(self, *args, **kwargs):
        raise NotImplementedError


class CrossAttentionEncoder(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 feature_dim: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + feature_dim, width, device=device, dtype=dtype)
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """
        bs = pc.shape[0]

        data = self.fourier_embedder(pc)
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        query = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc


class CrossAttentionDecoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 out_channels: int,
                 fourier_embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.fourier_embedder = fourier_embedder

        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)

        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash
        )

        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        queries = self.query_proj(self.fourier_embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x


class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.num_latents = num_latents
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )

        self.embed_dim = embed_dim
        if embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = nn.Linear(embed_dim, width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
        else:
            self.latent_shape = (num_latents, width)

        self.transformer = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )

        # geometry decoder
        self.geo_decoder = CrossAttentionDecoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            sample_posterior (bool):

        Returns:
            latents (torch.FloatTensor)
            center_pos (torch.FloatTensor or None):
            posterior (DiagonalGaussianDistribution or None):
        """

        latents, center_pos = self.encoder(pc, feats)

        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                latents = posterior.sample()
            else:
                latents = posterior.mode()

        return latents, center_pos, posterior

    def decode(self, latents: torch.FloatTensor):
        latents = self.post_kl(latents)
        return self.transformer(latents)

    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = self.geo_decoder(queries, latents).squeeze(-1)
        return logits

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            volume_queries (torch.FloatTensor): [B, P, 3]
            sample_posterior (bool):

        Returns:
            logits (torch.FloatTensor): [B, P]
            center_pos (torch.FloatTensor): [B, M, 3]
            posterior (DiagonalGaussianDistribution or None).

        """

        latents, center_pos, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)

        latents = self.decode(latents)
        logits = self.query_geometry(volume_queries, latents)

        return logits, center_pos, posterior

    # def training_step(self, batch):
        

class SurfaceFieldAutoEncoder(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype] = torch.float32,
                 num_latents,
                 feature_dim,
                 embed_dim,
                 num_freqs: int=8,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 init_scale=0.25,
                 ):
        super().__init__()

        self.num_latents = num_latents
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=True)

        init_scale = init_scale * math.sqrt(1.0 / width)
        self.embed_dim = embed_dim
        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            feature_dim=feature_dim,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=True,
            flash=False,
            use_ln_post=False,
            use_checkpoint=False
        )
        self.pre_kl = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
        self.post_kl = nn.Linear(embed_dim, width, device=device, dtype=dtype)
        self.latent_shape = (num_latents, embed_dim)

        self.transformer = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=True,
            flash=False,
            use_checkpoint=False
        )
        # geometry decoder
        self.geo_decoder = CrossAttentionDecoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=True,
            flash=False,
            use_checkpoint=False
        )
    
    def encode(self, coords, feats=None, sample_posterior=True):
        latents, center_pos = self.encoder(coords, feats)
        moments = self.pre_kl(latents)
        posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        latents = posterior.sample()
        return latents, center_pos, posterior
    
    def decode(self, latents: torch.FloatTensor):
        latents = self.post_kl(latents)
        return self.transformer(latents)
    
    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        feats = self.geo_decoder(queries, latents).squeeze(-1)
        return feats

    def forward(self,
                coords: torch.FloatTensor,
                feats: torch.FloatTensor,
                queries: torch.FloatTensor,
                sample_posterior: bool = True):
        latents, center_pos, posterior = self.encode(coords, feats, sample_posterior=sample_posterior)
        latents = self.decode(latents)
        recon_feats = self.query_geometry(queries, latents)
        if sample_posterior:
            kl_loss = posterior.kl(dims=(1,2))
            kl_loss = torch.mean(kl_loss)
        else:
            kl_loss = torch.tensor(0.0, device=recon_feats.device)
        return recon_feats, center_pos, kl_loss
    


"""


conda create -n test python=3.8

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

unzip checkpoints.zip


"""