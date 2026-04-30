#!/usr/bin/env python3
# evaluate.py
# ──────────────────────────────────────────────────────────────
# Unified evaluation script for the Arabic NLP Benchmark.
#
# Usage:
#   python evaluate.py --config-name sentiment
#   python evaluate.py --config-name ner
#   python evaluate.py --config-name sentiment dataset.max_samples=500
#
# Outputs:
#   - results/sentiment_results.csv   (per-model scores)
#   - results/plots/                  (bar charts + confusion matrices)
#   - Console table with all metrics
# ──────────────────────────────────────────────────────────────

from __future__ import annotations
import os, sys, time, importlib, logging
from pathlib import Path
from typing import Dict, List, Any

import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# ── Metrics ───────────────────────────────────────────────────
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ── Plotting ──────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# Dataset loaders
# ══════════════════════════════════════════════════════════════

def load_dataset(cfg: DictConfig):
    """Load the configured dataset and return (texts, labels)."""
    from datasets import load_dataset as hf_load

    name = cfg.dataset.name
    split = cfg.dataset.split
    text_col = cfg.dataset.text_column
    label_col = cfg.dataset.label_column
    max_samples = cfg.dataset.max_samples

    log.info(f"Loading dataset '{name}' (split={split}) …")

    if name == "hard":
        ds = hf_load("hard", split=split, trust_remote_code=True)
    elif name == "arsas":
        ds = hf_load("arbml/arsas", split=split, trust_remote_code=True)
    elif name == "anercorp":
        # Local file fallback
        data_path = Path("data/anercorp/anercorp.txt")
        if not data_path.exists():
            raise FileNotFoundError(
                "ANERcorp not found. Run data/download.sh first.\n"
                "Download manually from: https://camel.abudhabi.nyu.edu/anercorp/"
            )
        ds = hf_load("text", data_files=str(data_path), split=split)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    texts  = ds[text_col]
    labels = ds[label_col]

    log.info(f"  → {len(texts)} samples loaded")
    return texts, labels


# ══════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════

def build_model(model_cfg: DictConfig, task: str):
    """Instantiate a model from its config entry."""
    cls_name = model_cfg["class"]
    kwargs   = dict(model_cfg.get("kwargs", {}))
    kwargs["task"] = task

    # Dynamically import the right module
    module_map = {
        "AraBERTModel":          "models.arabert",
        "CAMeLBERTModel":        "models.camelbert",
        "MARBERTModel":          "models.marbert",
        "MultilingualBERTModel": "models.multilingual_bert",
    }
    module = importlib.import_module(module_map[cls_name])
    cls    = getattr(module, cls_name)
    return cls(**kwargs)


# ══════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════

def compute_metrics(y_true: List, y_pred: List, task: str) -> Dict[str, float]:
    """Compute all relevant metrics for the given task."""
    metrics = {}

    if task == "sentiment":
        metrics["accuracy"]    = accuracy_score(y_true, y_pred)
        metrics["f1_macro"]    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        metrics["precision"]   = precision_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["recall"]      = recall_score(y_true, y_pred, average="macro",    zero_division=0)

    elif task == "ner":
        metrics["f1_micro"]    = f1_score(y_true, y_pred, average="micro",    zero_division=0)
        metrics["f1_macro"]    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        metrics["precision"]   = precision_score(y_true, y_pred, average="micro", zero_division=0)
        metrics["recall"]      = recall_score(y_true, y_pred, average="micro",    zero_division=0)

    return {k: round(v * 100, 2) for k, v in metrics.items()}


def extract_labels_sentiment(predictions: List[Any]) -> List[int]:
    """Map HuggingFace text-classification output → integer labels."""
    label_map = {
        "LABEL_0": 0, "LABEL_1": 1,
        "NEGATIVE": 0, "POSITIVE": 1,
        "NEG": 0, "POS": 1,
    }
    preds = []
    for p in predictions:
        raw = p["label"].upper()
        preds.append(label_map.get(raw, int(raw.split("_")[-1])))
    return preds


# ══════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════

def plot_comparison_bar(results_df: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """Bar chart comparing all models on a single metric."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(results_df)))
    bars = ax.barh(results_df["model"], results_df[metric], color=colors)
    ax.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    ax.set_xlabel(f"{metric} (%)", fontsize=11)
    ax.set_title(f"Arabic NLP Benchmark — {metric}", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"bar_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved plot → {path}")


def plot_confusion_matrix(y_true: List, y_pred: List, model_name: str, output_dir: Path) -> None:
    """Confusion matrix heatmap for a single model."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11)
    plt.tight_layout()
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    path = output_dir / f"cm_{safe_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

@hydra.main(config_path="configs", config_name="sentiment", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Config:\n" + OmegaConf.to_yaml(cfg))

    # ── Setup ─────────────────────────────────────────────────
    results_dir = Path(cfg.output.results_dir)
    plots_dir   = Path(cfg.output.plots_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Optional W&B
    if cfg.output.log_wandb:
        import wandb
        wandb.init(project=cfg.output.wandb_project, config=OmegaConf.to_container(cfg))

    # ── Load data ─────────────────────────────────────────────
    texts, labels = load_dataset(cfg)

    # ── Run each model ────────────────────────────────────────
    all_results = []
    all_preds   = {}   # model_name → predictions (for confusion matrices)

    enabled_models = {k: v for k, v in cfg.models.items() if v.get("enabled", True)}
    log.info(f"\nRunning {len(enabled_models)} models …\n{'─'*50}")

    for model_key, model_cfg in enabled_models.items():
        log.info(f"\n▶ {model_key}")
        model = build_model(model_cfg, cfg.task)
        model.load()

        # Inference
        t0 = time.time()
        raw_preds = model.predict(list(texts))
        elapsed = time.time() - t0

        # Post-process predictions
        if cfg.task == "sentiment":
            y_pred = extract_labels_sentiment(raw_preds)
        else:
            y_pred = raw_preds   # task-specific post-processing

        y_true = list(labels)

        # Metrics
        metrics = compute_metrics(y_true, y_pred, cfg.task)
        metrics["model"]        = model_key
        metrics["inference_sec"] = round(elapsed, 1)
        metrics["samples_per_sec"] = round(len(texts) / elapsed, 0)

        all_results.append(metrics)
        all_preds[model_key] = (y_true, y_pred)

        log.info(f"  {metrics}")

        # Confusion matrix
        if cfg.task == "sentiment":
            plot_confusion_matrix(y_true, y_pred, model_key, plots_dir)

        # Free VRAM
        del model

    # ── Aggregate results ─────────────────────────────────────
    df = pd.DataFrame(all_results)
    col_order = ["model"] + [c for c in df.columns if c != "model"]
    df = df[col_order].sort_values("f1_macro" if "f1_macro" in df.columns else df.columns[1], ascending=False)

    # Save CSV
    df.to_csv(cfg.output.csv_file, index=False)
    log.info(f"\nResults saved → {cfg.output.csv_file}")

    # Pretty print
    print("\n" + "═"*70)
    print(f"  ARABIC NLP BENCHMARK — {cfg.task.upper()} RESULTS")
    print("═"*70)
    print(df.to_string(index=False))
    print("═"*70 + "\n")

    # ── Plots ─────────────────────────────────────────────────
    primary_metric = "f1_macro" if "f1_macro" in df.columns else "accuracy"
    plot_comparison_bar(df, primary_metric, plots_dir)
    if "accuracy" in df.columns:
        plot_comparison_bar(df, "accuracy", plots_dir)

    # ── W&B summary ───────────────────────────────────────────
    if cfg.output.log_wandb:
        import wandb
        wandb.log({"results_table": wandb.Table(dataframe=df)})
        wandb.finish()

    log.info("Done ✓")


if __name__ == "__main__":
    main()
