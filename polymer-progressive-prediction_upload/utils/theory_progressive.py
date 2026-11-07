from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse

import json

import math

import os

import random

import shutil

from typing import List, Optional

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from torch.optim import lr_scheduler

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

PAD_ID = 0

UNK_ID = 1

CH1_MAX_LEN = 128

CH2_MAX_LEN = 128

CH3_MAX_LEN = 128

def set_seed(seed: int = 992603):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.PReLU):
        torch.nn.init.constant_(m.weight, 0.25)

def set_journal_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'svg.fonttype': 'none',
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

set_journal_style()

DATASET_DIR = str(PROJECT_ROOT / "data/theory")

def eval_and_plot(y_true: np.ndarray, y_pred: np.ndarray, smiles: np.ndarray, path: str, task: str = r"$\langle R^{2} \rangle_{0}$ (Å²)"):

    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    smiles = np.array(smiles).reshape(-1)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (y_true > 0) & (y_pred > 0)
    y_t, y_p = y_true[mask], y_pred[mask]

    if len(y_t) == 0:
        print("Warning: No valid data for plotting!")
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)

    fig, ax = plt.subplots(1, 1, figsize=(3.3, 3.2), dpi=600)

    scatter_base = {'s': 16, 'alpha': 0.5, 'edgecolors': 'white',
                    'linewidth': 0.5}

    ax.scatter(y_t, y_p, c='#0F4D92', **scatter_base, zorder=5)

    lim_min = y_t.min() * 0.8
    lim_max = y_t.max() * 1.2
    lims = np.logspace(np.log10(lim_min), np.log10(lim_max), 100)
    ax.plot(lims, lims, color='black', lw=1.0, alpha=0.9, zorder=2)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("True " + task)
    ax.set_ylabel("Predicted " + task)

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    text = f"MAE = {mae:.1f}\nRMSE = {rmse:.1f}\n$R^2$ = {r2:.4f}"
    ax.text(0.96, 0.04, text, transform=ax.transAxes, fontsize=7,
            va='bottom', ha='right',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92,
                      edgecolor="none"))

    plt.tight_layout(pad=0.6)
    plt.savefig(f"{path}.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(f"{path}.png", dpi=600, bbox_inches='tight', transparent=False)
    plt.savefig(f"{path}.tiff", dpi=600, bbox_inches='tight', format='tiff')

    plt.close()

    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}
