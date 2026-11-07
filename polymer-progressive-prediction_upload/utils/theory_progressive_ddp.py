from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse

import json

import math

import os

import random

import shutil

import sys

from typing import List, Optional

import numpy as np

import pandas as pd

import torch

import torch.distributed as dist

import torch.nn as nn

import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP

from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, Dataset

from torch.utils.data.distributed import DistributedSampler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PAD_ID = 0

UNK_ID = 1

CH1_MAX_LEN = 128

CH2_MAX_LEN = 128

CH3_MAX_LEN = 128

DATASET_DIR = str(PROJECT_ROOT / "data/theory")

def set_seed(seed: int = 992603):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.PReLU):
        nn.init.constant_(m.weight, 0.25)

def is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0

def eval_plot(y_true, y_pred, path: str):
    y_t = np.asarray(y_true).reshape(-1)
    y_p = np.asarray(y_pred).reshape(-1)
    mask = (~np.isnan(y_t)) & (~np.isnan(y_p)) & (y_t > 0) & (y_p > 0)
    y_t, y_p = y_t[mask], y_p[mask]

    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)

    fig, ax = plt.subplots(figsize=(3.3, 3.2), dpi=600)
    ax.scatter(y_t, y_p, c="#0F4D92", s=16, alpha=0.5, edgecolors="white", linewidth=0.5, zorder=5)
    lo, hi = y_t.min() * 0.8, y_t.max() * 1.2
    lims = np.logspace(np.log10(lo), np.log10(hi), 100)
    ax.plot(lims, lims, color="black", lw=1.0, alpha=0.9, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"True $\langle R^2 \rangle_0$ (Å²)")
    ax.set_ylabel(r"Predicted $\langle R^2 \rangle_0$ (Å²)")
    ax.text(
        0.96, 0.04, f"MAE = {mae:.1f}\nRMSE = {rmse:.1f}\n$R^2$ = {r2:.4f}",
        transform=ax.transAxes, fontsize=7, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor="none"),
    )
    for ext in ["pdf", "png", "tiff"]:
        fig.savefig(f"{path}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
