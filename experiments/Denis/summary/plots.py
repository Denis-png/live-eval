"""
plots.py — Comparison plots: original FCE vs generated benchmark across GEC models.

Run from Denis/:
    python summary/plots.py

Input:  data/summary/summary_comparison.csv
Output: data/summary/summary_f05_bar.png
        data/summary/summary_cola_bar.png
        data/summary/summary_correction_bar.png
        data/summary/summary_gt_metrics_bar.png
        data/summary/summary_heatmap.png
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

ROOT    = Path(__file__).parent.parent
SUMMARY = ROOT / 'data/summary'
COMP    = SUMMARY / 'summary_comparison.csv'

MODEL_LABELS = {
    't5_base':      'T5-base',
    'coedit_large': 'CoEdit-L',
    'prithivida':   'Prithivida',
    'haiku':        'Haiku 4.5',
}
DATASET_COLORS = {
    'original':  '#4C72B0',
    'generated': '#DD8452',
}
MODEL_ORDER = ['t5_base', 'coedit_large', 'prithivida', 'haiku']


def load() -> pd.DataFrame:
    if not COMP.exists():
        raise FileNotFoundError(f'{COMP} not found — run evaluation.py first.')
    df = pd.read_csv(COMP)
    # Enforce model order
    df['model'] = pd.Categorical(df['model'], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(['model', 'dataset']).reset_index(drop=True)


def _grouped_bar(df: pd.DataFrame, metric: str, ylabel: str, title: str,
                 out_name: str, ylim: tuple = (0, 1.05)) -> None:
    models   = [m for m in MODEL_ORDER if m in df['model'].values]
    datasets = ['original', 'generated']
    x        = np.arange(len(models))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, ds in enumerate(datasets):
        vals = [
            df.loc[(df['model'] == m) & (df['dataset'] == ds), metric].values[0]
            if len(df.loc[(df['model'] == m) & (df['dataset'] == ds)]) > 0 else 0.0
            for m in models
        ]
        bars = ax.bar(
            x + (i - 0.5) * width, vals, width,
            label=ds.capitalize(), color=DATASET_COLORS[ds], alpha=0.85,
        )
        ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    out = SUMMARY / out_name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 1: ERRANT F0.5 ───────────────────────────────────────────────────────

def plot_f05(df: pd.DataFrame) -> None:
    _grouped_bar(
        df, 'errant_f0.5',
        ylabel='ERRANT F0.5',
        title='ERRANT F0.5 — GEC model × dataset (vs FCE reference distribution)',
        out_name='summary_f05_bar.png',
    )


# ── Plot 2: CoLA input vs corrected ──────────────────────────────────────────

def plot_cola(df: pd.DataFrame) -> None:
    """Show cola_input and cola_corrected side by side, per model, one subplot per dataset."""
    models   = [m for m in MODEL_ORDER if m in df['model'].values]
    datasets = ['original', 'generated']
    x        = np.arange(len(models))
    width    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, ds in zip(axes, datasets):
        sub = df[df['dataset'] == ds]
        inp_vals  = [sub.loc[sub['model'] == m, 'cola_input'].values[0]     if len(sub[sub['model'] == m]) > 0 else 0 for m in models]
        corr_vals = [sub.loc[sub['model'] == m, 'cola_corrected'].values[0] if len(sub[sub['model'] == m]) > 0 else 0 for m in models]

        b1 = ax.bar(x - width/2, inp_vals,  width, label='Input',     color='#C44E52', alpha=0.85)
        b2 = ax.bar(x + width/2, corr_vals, width, label='Corrected', color='#55A868', alpha=0.85)
        ax.bar_label(b1, fmt='%.2f', padding=2, fontsize=7)
        ax.bar_label(b2, fmt='%.2f', padding=2, fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
        ax.set_title(f'{ds.capitalize()} FCE' if ds == 'original' else 'Generated benchmark')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    axes[0].set_ylabel('CoLA acceptability rate')
    fig.suptitle('CoLA: input vs GEC-corrected (original FCE vs generated benchmark)', fontsize=12)
    fig.tight_layout()
    out = SUMMARY / 'summary_cola_bar.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 3: Correction extent + n_edits ──────────────────────────────────────

def plot_correction(df: pd.DataFrame) -> None:
    models   = [m for m in MODEL_ORDER if m in df['model'].values]
    datasets = ['original', 'generated']
    x        = np.arange(len(models))
    width    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, ylabel, title_sfx in zip(
        axes,
        ['correction_extent', 'n_edits'],
        ['1 − bigram Jaccard', 'Mean ERRANT edit count'],
        ['Bigram-level text change', 'ERRANT edit count per sentence'],
    ):
        for i, ds in enumerate(datasets):
            vals = [
                df.loc[(df['model'] == m) & (df['dataset'] == ds), metric].values[0]
                if len(df.loc[(df['model'] == m) & (df['dataset'] == ds)]) > 0 else 0
                for m in models
            ]
            bars = ax.bar(
                x + (i - 0.5) * width, vals, width,
                label=ds.capitalize(), color=DATASET_COLORS[ds], alpha=0.85,
            )
            ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(title_sfx)
        ax.legend(fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle('Correction depth — original FCE vs generated benchmark', fontsize=12)
    fig.tight_layout()
    out = SUMMARY / 'summary_correction_bar.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 4: Ground-truth metrics (errant_f0.5_gt + gleu_gt) ──────────────────

def plot_gt_metrics(df: pd.DataFrame) -> None:
    """Side-by-side bar charts for the two per-sentence GT-based metrics."""
    available = [c for c in ['errant_f0.5_gt', 'gleu_gt'] if c in df.columns]
    if not available:
        print('No GT metrics found — skipping plot_gt_metrics.')
        return

    models   = [m for m in MODEL_ORDER if m in df['model'].values]
    datasets = ['original', 'generated']
    x        = np.arange(len(models))
    width    = 0.35

    ylabels = {'errant_f0.5_gt': 'ERRANT F0.5 (sentence-level GT)',
               'gleu_gt':        'GLEU (sentence-level GT)'}
    titles  = {'errant_f0.5_gt': 'ERRANT F0.5 — per-sentence ground truth reference',
               'gleu_gt':        'GLEU — per-sentence ground truth reference'}

    fig, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 5),
                             sharey=False)
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        for i, ds in enumerate(datasets):
            vals = [
                df.loc[(df['model'] == m) & (df['dataset'] == ds), metric].values[0]
                if len(df.loc[(df['model'] == m) & (df['dataset'] == ds)]) > 0 else 0.0
                for m in models
            ]
            bars = ax.bar(
                x + (i - 0.5) * width, vals, width,
                label=ds.capitalize(), color=DATASET_COLORS[ds], alpha=0.85,
            )
            ax.bar_label(bars, fmt='%.3f', padding=2, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=11)
        ax.set_ylabel(ylabels[metric])
        ax.set_title(titles[metric])
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    fig.suptitle('GT-referenced metrics — original FCE vs generated benchmark', fontsize=12)
    fig.tight_layout()
    out = SUMMARY / 'summary_gt_metrics_bar.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 5: Heatmap (all metrics) ─────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame) -> None:
    metrics = [c for c in ['errant_f0.5', 'errant_f0.5_gt', 'gleu_gt',
                            'cola_input', 'cola_corrected', 'cola_delta',
                            'correction_extent', 'n_edits'] if c in df.columns]
    models   = [m for m in MODEL_ORDER if m in df['model'].values]
    datasets = ['original', 'generated']

    row_labels = [f'{MODEL_LABELS.get(m, m)} / {ds}' for m in models for ds in datasets]
    data_rows  = []
    for m in models:
        for ds in datasets:
            sub = df[(df['model'] == m) & (df['dataset'] == ds)]
            if len(sub) == 0:
                data_rows.append([0.0] * len(metrics))
            else:
                data_rows.append([float(sub[c].values[0]) for c in metrics])

    data = np.array(data_rows)
    # Normalise each column to [0,1] for colour only (keep raw values as text)
    col_min = data.min(axis=0)
    col_max = data.max(axis=0)
    norm    = np.where(col_max > col_min, (data - col_min) / (col_max - col_min), 0.5)

    fig, ax = plt.subplots(figsize=(len(metrics) * 1.6, len(row_labels) * 0.55 + 1.5))
    im = ax.imshow(norm, aspect='auto', cmap='YlGn', vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=9, rotation=30, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title('Summary metric heatmap — GEC model × dataset (colour = column-normalised)')

    for i in range(len(row_labels)):
        for j in range(len(metrics)):
            ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center',
                    fontsize=8, color='black' if norm[i, j] < 0.6 else 'white')

    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.04, label='column-normalised')
    fig.tight_layout()
    out = SUMMARY / 'summary_heatmap.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


def main() -> None:
    df = load()
    plot_f05(df)
    plot_cola(df)
    plot_correction(df)
    plot_gt_metrics(df)
    plot_heatmap(df)


if __name__ == '__main__':
    main()
