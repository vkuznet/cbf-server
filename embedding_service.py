#!/usr/bin/env python3

import os
import numpy as np
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


# =========================
# Configuration
# =========================

EMBED_DIM = int(os.getenv("EMBED_DIM", "512"))
RESNET_VARIANT = int(os.getenv("RESNET_VARIANT", "18"))
DEVICE = os.getenv("DEVICE", "cpu")
IMAGE_SIZE = 224


# =========================
# Model
# =========================

class ResNetEmbedder(nn.Module):
    def __init__(self, embed_dim: int, variant: int):
        super().__init__()

        if variant == 18:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            dim = 512
        elif variant == 34:
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            dim = 512
        elif variant == 50:
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            dim = 2048
        else:
            raise ValueError("Unsupported ResNet variant")

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Linear(dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.projection(x)
        return nn.functional.normalize(x, dim=1)


model = ResNetEmbedder(EMBED_DIM, RESNET_VARIANT).to(DEVICE)
model.eval()


# =========================
# Input schema
# =========================

class PixelArrayRequest(BaseModel):
    pixels: List[float]          # flattened
    height: int
    width: int
    dtype: str = "float32"


# =========================
# Preprocessing
# =========================

resize_norm = T.Compose([
    T.ToPILImage(),
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def preprocess_pixels(pixels, h, w):
    arr = np.asarray(pixels, dtype=np.float32)

    if arr.size != h * w:
        raise ValueError("Pixel count does not match height × width")

    arr = arr.reshape(h, w)

    # Scientific preprocessing
    arr = np.log1p(arr)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()

    # Convert to 3-channel
    arr = np.stack([arr, arr, arr], axis=-1)

    tensor = resize_norm(arr).unsqueeze(0)
    return tensor


# =========================
# FastAPI app
# =========================

app = FastAPI(title="Pixel Embedding Service")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "embed_dim": EMBED_DIM,
        "resnet": RESNET_VARIANT,
        "device": DEVICE,
    }


@app.post("/embed/pixels")
def embed_pixels(req: PixelArrayRequest):
    try:
        x = preprocess_pixels(req.pixels, req.height, req.width)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    with torch.no_grad():
        emb = model(x.to(DEVICE))[0].cpu().numpy()

    return JSONResponse({
        "embedding": emb.tolist(),
        "dim": len(emb),
    })
