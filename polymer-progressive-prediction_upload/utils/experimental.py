from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

import argparse

import json

import math

import os

import shutil

from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.group_split import build_group_split_plan, save_group_split_plan

from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

import random

PAD_ID = 0

UNK_ID = 1

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

def set_seed(seed: int = 42):
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

def eval_and_plot(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    save_path: str,
    fold: int
):
    tasks = ["Mw", "viscosity", "Glass transition temperature"]
    labels = {
        "Mw": r"$M_{\mathrm{w}}$ (log scale)",
        "viscosity": r"Melt viscosity (log scale)",
        "Glass transition temperature": r"$T_{\mathrm{g}}$ (log scale)"
    }

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), dpi=600)

    scatter_kwargs = {
        's': 16,
        'alpha': 0.5,
        'c': '#0F4D92',
        'edgecolors': 'white',
        'linewidth': 0.5,
        'zorder': 5
    }

    metrics = {}

    for ax, task in zip(axs, tasks):
        y_t = np.asarray(y_true_dict[task]).ravel()
        y_p = np.asarray(y_pred_dict[task]).ravel()

        mask = (
            ~np.isnan(y_t) &
            ~np.isnan(y_p) &
            ~np.isinf(y_t) &
            ~np.isinf(y_p)
        )
        y_t, y_p = y_t[mask], y_p[mask]

        if len(y_t) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=7)
            metrics[task] = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
            continue

        mae  = mean_absolute_error(y_t, y_p)
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        r2   = r2_score(y_t, y_p)

        metrics[task] = {
            "MAE":  float(mae),
            "RMSE": float(rmse),
            "R2":   float(r2)
        }

        # 散点图
        ax.scatter(y_t, y_p, **scatter_kwargs)

        # 对角线（线性坐标）
        lim_min = min(y_t.min(), y_p.min()) * 0.95
        lim_max = max(y_t.max(), y_p.max()) * 1.05
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color='black', lw=1.0, alpha=0.9, zorder=2)

        ax.set_xlabel(f"True {labels[task]}")
        ax.set_ylabel(f"Predicted {labels[task]}")

        # 统计信息框（右下角，与你之前代码一致）
        text = f"MAE = {mae:.1f}\nRMSE = {rmse:.1f}\n$R^2$ = {r2:.4f}"
        ax.text(0.96, 0.04, text,
                transform=ax.transAxes,
                fontsize=7,
                va='bottom',
                ha='right',
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white",
                          alpha=0.92,
                          edgecolor="none"))

    plt.tight_layout(pad=0.6, w_pad=2.8)

    plt.savefig(f"{save_path}.pdf", dpi=600, bbox_inches='tight')
    plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight', transparent=False)
    plt.savefig(f"{save_path}.tiff", dpi=600, bbox_inches='tight', format='tiff')

    plt.close()

    # 打印本次 fold 的结果，便于观察
    print(f"\nFold {fold} Test Metrics (on log-transformed values):")
    for task in tasks:
        m = metrics[task]
        if not np.isnan(m["MAE"]):
            print(f"  {task:25s} | MAE: {m['MAE']:8.2f}   RMSE: {m['RMSE']:8.2f}   R²: {m['R2']:8.4f}")
        else:
            print(f"  {task:25s} | No valid data")

    return metrics

def plot_cv_summary(fold_summaries, out_dir):
    """
    3×3 summary bar chart of 5-fold CV results.
    Rows: MAE, RMSE, R² — Columns: Mw, Melt viscosity, Glass transition temperature.
    Each bar = one fold. Nature journal style.
    """
    tasks = ["Mw", "viscosity", "Glass transition temperature"]
    task_labels = ["Mw", "Melt viscosity", r"$T_{\mathrm{g}}$"]
    metrics = ["MAE", "RMSE", "R2"]
    metric_labels = ["MAE", "RMSE", r"$R^2$"]

    # 期刊友好低饱和度配色
    task_colors = {
        "Mw": "#8cacc6",                         # 浅蓝灰
        "viscosity": "#d9b38c",                  # 浅橙灰
        "Glass transition temperature": "#a0c49d",  # 浅绿灰
    }

    n_folds = len(fold_summaries)

    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.5), dpi=600)

    for row, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        for col, (task, tlabel) in enumerate(zip(tasks, task_labels)):
            ax = axes[row, col]

            values = [fold_summaries[f][task][metric] for f in range(n_folds)]

            x = np.arange(n_folds)
            ax.bar(x, values,
                   color=task_colors[task],
                   edgecolor='#272727',
                   linewidth=0.6,
                   alpha=0.92,
                   width=0.65)

            ax.set_xticks(x)
            ax.set_xticklabels([f'Fold {i + 1}' for i in range(n_folds)])

            # column titles on top row
            if row == 0:
                ax.set_title(tlabel, fontsize=8, pad=10)

            # metric labels on leftmost column
            if col == 0:
                ax.set_ylabel(mlabel)

            # 全框图：四边脊柱均可见
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
            ax.tick_params(width=0.5)
            ax.grid(False)

    plt.tight_layout(pad=0.6, h_pad=1.5, w_pad=1.2)

    summary_path = os.path.join(out_dir, "cv_summary_3x3")
    fig.savefig(f"{summary_path}.pdf", dpi=600, bbox_inches='tight')
    fig.savefig(f"{summary_path}.png", dpi=600, bbox_inches='tight', transparent=False)
    fig.savefig(f"{summary_path}.tiff", dpi=600, bbox_inches='tight', format='tiff')
    plt.close(fig)

    print(f"CV summary figure saved to {summary_path}.pdf / .png")
