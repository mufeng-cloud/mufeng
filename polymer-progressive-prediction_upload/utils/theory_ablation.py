from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse

import json

import os

import random

from typing import List, Optional

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import glob

import matplotlib.pyplot as plt

import glob

import shutil

from rdkit import Chem

from rdkit.Chem import Descriptors

PAD_ID = 0

UNK_ID = 1

CH1_MAX_LEN = 128

CH2_MAX_LEN = 2500

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

def eval_and_plot(y_true_re: np.ndarray, y_pred_re: np.ndarray,
                  smiles: np.ndarray, path: str):
    """
    绘制单一 ⟨R²⟩₀ 的散点图（log-log 对数坐标），
    风格与顶级期刊（如 Macromolecules、ACS Macro Letters）完全一致。
    """
    y_true_re = np.array(y_true_re).reshape(-1)
    y_pred_re = np.array(y_pred_re).reshape(-1)
    smiles = np.array(smiles).reshape(-1)

    mask_re = ~np.isnan(y_true_re) & ~np.isnan(y_pred_re) & (y_true_re > 0) & (y_pred_re > 0)
    y_t_re, y_p_re = y_true_re[mask_re], y_pred_re[mask_re]

    if len(y_t_re) == 0:
        print("Warning: No valid positive data for log-scale plotting!")
        return {"Re": {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}}

    mae_re = mean_absolute_error(y_t_re, y_p_re)
    rmse_re = np.sqrt(mean_squared_error(y_t_re, y_p_re))
    r2_re = r2_score(y_t_re, y_p_re)

    fig, ax = plt.subplots(1, 1, figsize=(3.3, 3.2), dpi=600)

    scatter_base = {'s': 16, 'alpha': 0.5, 'edgecolors': 'white',
                    'linewidth': 0.5}

    ax.scatter(y_t_re, y_p_re, c='#0F4D92', **scatter_base, zorder=5)

    ax.set_xscale('log')
    ax.set_yscale('log')

    lim_min_re = y_t_re.min() * 0.8
    lim_max_re = y_t_re.max() * 1.2
    lims_re = np.logspace(np.log10(lim_min_re), np.log10(lim_max_re), 100)
    ax.plot(lims_re, lims_re, color='black', lw=1.0, alpha=0.9, zorder=2)

    ax.set_xlabel("True " + r"$\langle R^{2} \rangle_{0}$ (Å²)")
    ax.set_ylabel("Predicted " + r"$\langle R^{2} \rangle_{0}$ (Å²)")

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    text_re = f"MAE = {mae_re:.1f}\nRMSE = {rmse_re:.1f}\n$R^2$ = {r2_re:.4f}"
    ax.text(0.96, 0.04, text_re, transform=ax.transAxes, fontsize=7,
            va='bottom', ha='right',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92, edgecolor="none"))

    plt.tight_layout(pad=0.6)
    plt.savefig(f"{path}.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(f"{path}.png", dpi=600, bbox_inches='tight', transparent=False)
    plt.savefig(f"{path}.tiff", dpi=600, bbox_inches='tight', format='tiff')

    plt.close()

    return {"Re": {"MAE": float(mae_re), "RMSE": float(rmse_re), "R2": float(r2_re)}}

DATASET_DIR = str(PROJECT_ROOT / "data/theory")
