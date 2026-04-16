"""
plots.py — Descriptive plots for dataset tier comparison.

Run from Denis/datasets_experiments/:
    python plots.py

Input:  data/results/datasets_comparison.csv
        data/corrected/ds_{name}_corrected.csv  (for per-sentence distributions)
Output: data/results/ds_metrics_bar.png
        data/results/ds_cola_bar.png
        data/results/ds_jaccard_bar.png
        data/results/ds_error_type_dist.png     (ERRANT type breakdown per tier)
        data/results/ds_metrics_heatmap.png
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from collections import Counter

warnings.filterwarnings('ignore')

ROOT    = Path(__file__).parent.parent
RESULTS = ROOT / 'data/results'
COMP    = RESULTS / 'datasets_comparison.csv'

TIER_ORDER  = ['conll14', 'fce', 'c4_200m']
TIER_LABELS = {'conll14': 'CoNLL-14\n(Test set)', 'fce': 'FCE\n(Benchmark)', 'c4_200m': 'C4_200M\n(General)'}
PALETTE     = ['#4C72B0', '#DD8452', '#55A868']


def load_comparison() -> pd.DataFrame:
    if not COMP.exists():
        raise FileNotFoundError(f'{COMP} not found — run evaluation.py first.')
    df = pd.read_csv(COMP, index_col=0)
    order = [d for d in TIER_ORDER if d in df.index]
    return df.loc[order]


# ── Plot 1: multi-metric bar ──────────────────────────────────────────────────

def plot_metrics_bar(df: pd.DataFrame) -> None:
    metrics = ['errant_f0.5', 'cola_B', 'cola_C', 'bigram_jaccard']
    labels  = ['ERRANT F0.5', 'CoLA-B (generated)', 'CoLA-C (corrected)', 'Bigram Jaccard']
    x       = np.arange(len(df))
    width   = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = df[metric].values if metric in df.columns else np.zeros(len(df))
        ax.bar(x + i * width, vals, width, label=label, color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([TIER_LABELS.get(v, v) for v in df.index], fontsize=10)
    ax.set_ylabel('Score')
    ax.set_title('Dataset tier comparison — all metrics')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    out = RESULTS / 'ds_metrics_bar.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 2: CoLA A / B / C grouped bar ───────────────────────────────────────

def plot_cola_bar(df: pd.DataFrame) -> None:
    cola_cols = [c for c in ['cola_A', 'cola_B', 'cola_C'] if c in df.columns]
    cola_lbls = {'cola_A': 'Original (A)', 'cola_B': 'Generated (B)', 'cola_C': 'T5-corrected (C)'}
    x     = np.arange(len(df))
    width = 0.25
    colors = ['#4C72B0', '#C44E52', '#55A868']

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, col in enumerate(cola_cols):
        ax.bar(x + i * width, df[col].values, width,
               label=cola_lbls[col], color=colors[i], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([TIER_LABELS.get(v, v) for v in df.index], fontsize=10)
    ax.set_ylabel('CoLA acceptability rate')
    ax.set_title('CoLA scores: Original → Generated → T5-corrected')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    out = RESULTS / 'ds_cola_bar.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 3: Jaccard + memorisation rate ───────────────────────────────────────

def plot_jaccard_bar(df: pd.DataFrame) -> None:
    if 'bigram_jaccard' not in df.columns:
        return
    cols   = [c for c in ['bigram_jaccard', 'pct_memorised', 'corr_extent'] if c in df.columns]
    labels = {
        'bigram_jaccard': 'Novelty — orig→gen (lower = more novel)',
        'pct_memorised':  '% Jaccard > 0.3 (potential memorisation)',
        'corr_extent':    'Correction extent — gen→T5 (higher = more corrected)',
    }
    colors = ['#4C72B0', '#DD8452', '#55A868']
    x      = np.arange(len(df))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, col in enumerate(cols):
        offset = (i - len(cols) / 2 + 0.5) * width
        ax.bar(x + offset, df[col].values, width,
               label=labels[col], color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS.get(v, v) for v in df.index], fontsize=10)
    ax.set_ylabel('Score / proportion')
    ax.set_title('Bigram Jaccard — novelty (orig→gen) and correction extent (gen→T5)')
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    out = RESULTS / 'ds_jaccard_bar.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 4: ERRANT error-type distribution per tier ───────────────────────────

def plot_error_type_dist() -> None:
    """Read corrected CSVs to get ERRANT type distribution per tier."""
    try:
        import spacy, errant as errant_lib
        try:
            spacy.load('en_core_web_sm')
        except OSError:
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
        ann = errant_lib.load('en')
    except ImportError:
        print('errant not available — skipping error-type plot.')
        return

    tier_dists: dict[str, dict] = {}
    for name in TIER_ORDER:
        cor_path = ROOT / f'data/corrected/ds_{name}_corrected.csv'
        if not cor_path.exists():
            continue
        df = pd.read_csv(cor_path)
        edits = [
            ann.annotate(ann.parse(g), ann.parse(t))
            for g, t in zip(df['generated'].tolist(), df['t5_corrected'].tolist())
        ]
        c = Counter(e.type for edit_list in edits for e in edit_list)
        total = sum(c.values()) or 1
        tier_dists[name] = {k: v / total for k, v in c.most_common(8)}

    if not tier_dists:
        return

    all_types = list(dict.fromkeys(t for d in tier_dists.values() for t in d))[:8]
    x     = np.arange(len(all_types))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, dist) in enumerate(tier_dists.items()):
        vals = [dist.get(t, 0) for t in all_types]
        ax.bar(x + i * width, vals, width, label=TIER_LABELS.get(name, name),
               color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(all_types, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Proportion of edits')
    ax.set_title('ERRANT error-type distribution — hypothesis edits (B → C) per tier')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    out = RESULTS / 'ds_error_type_dist.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 5: heatmap ────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame) -> None:
    num_cols = [c for c in ['errant_f0.5', 'cola_A', 'cola_B', 'cola_C',
                             'bigram_jaccard', 'pct_memorised', 'corr_extent'] if c in df.columns]
    data    = df[num_cols].astype(float)
    row_lbl = [TIER_LABELS.get(v, v).replace('\n', ' ') for v in data.index]

    fig, ax = plt.subplots(figsize=(len(num_cols) * 1.4, len(data) * 0.8 + 1.5))
    im = ax.imshow(data.values, aspect='auto', cmap='YlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(len(row_lbl)))
    ax.set_yticklabels(row_lbl, fontsize=10)
    ax.set_title('Dataset tier metric heatmap')
    for i in range(len(data)):
        for j, col in enumerate(num_cols):
            val = data.values[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9,
                    color='black' if val < 0.7 else 'white')
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    fig.tight_layout()
    out = RESULTS / 'ds_metrics_heatmap.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


def main() -> None:
    df = load_comparison()
    plot_metrics_bar(df)
    plot_cola_bar(df)
    plot_jaccard_bar(df)
    plot_heatmap(df)
    plot_error_type_dist()


if __name__ == '__main__':
    main()
