"""
plots.py — Descriptive plots for prompt engineering results.

Run from Denis/prompt_engineering/:
    python plots.py

Input:  data/results/prompt_leaderboard.csv
        data/results/ablation_leaderboard.csv  (optional)
Output: data/results/pe_metrics_bar.png
        data/results/pe_metrics_heatmap.png
        data/results/pe_ablation_bar.png        (if ablation file exists)
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT      = Path(__file__).parent.parent
RESULTS   = ROOT / 'data/results'
LEADER    = RESULTS / 'prompt_leaderboard.csv'
ABLATION  = RESULTS / 'ablation_leaderboard.csv'

VARIANT_ORDER = ['zero_shot', 'one_shot', 'few_shot_3', 'few_shot_12', 'cot']
VARIANT_LABELS = {
    'zero_shot':   'Zero-shot',
    'one_shot':    'One-shot',
    'few_shot_3':  'Few-shot 3',
    'few_shot_12': 'Few-shot 12',
    'cot':         'CoT',
}

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']


def load_leaderboard() -> pd.DataFrame:
    if not LEADER.exists():
        raise FileNotFoundError(f'{LEADER} not found — run evaluation.py first.')
    df = pd.read_csv(LEADER, index_col=0)
    # Reorder rows by VARIANT_ORDER (keep only known variants)
    order = [v for v in VARIANT_ORDER if v in df.index]
    return df.loc[order]


# ── Plot 1: side-by-side metric bars ──────────────────────────────────────────

def plot_metrics_bar(df: pd.DataFrame) -> None:
    metrics  = ['f0.5', 'cola_B', 'cola_C', 'sem_sim', 'corr_extent']
    labels   = ['ERRANT F0.5', 'CoLA-B (generated)', 'CoLA-C (corrected)',
                'Novelty (orig→gen)', 'Correction extent (gen→T5)']
    x        = np.arange(len(df))
    width    = 0.15
    fig, ax  = plt.subplots(figsize=(10, 5))

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        vals = df[metric].values if metric in df.columns else np.zeros(len(df))
        ax.bar(x + i * width, vals, width, label=label, color=PALETTE[i], alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in df.index], fontsize=10)
    ax.set_ylabel('Score')
    ax.set_title('Prompt variant comparison — all metrics')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    out = RESULTS / 'pe_metrics_bar.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 2: heatmap of all metrics ────────────────────────────────────────────

def plot_metrics_heatmap(df: pd.DataFrame) -> None:
    cols    = [c for c in ['f0.5', 'n_edits', 'cola_A', 'cola_B', 'cola_C',
                           'sem_sim', 'corr_extent'] if c in df.columns]
    data    = df[cols].astype(float)
    row_lbl = [VARIANT_LABELS.get(v, v) for v in data.index]

    fig, ax = plt.subplots(figsize=(len(cols) * 1.3, len(data) * 0.7 + 1.2))
    im = ax.imshow(data.values, aspect='auto', cmap='YlGn', vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=10, rotation=30, ha='right')
    ax.set_yticks(range(len(row_lbl)))
    ax.set_yticklabels(row_lbl, fontsize=10)
    ax.set_title('Metric heatmap — prompt variants')

    for i in range(len(data)):
        for j, col in enumerate(cols):
            val = data.values[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9,
                    color='black' if val < 0.7 else 'white')

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    fig.tight_layout()
    out = RESULTS / 'pe_metrics_heatmap.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 3: ablation F0.5 bar ─────────────────────────────────────────────────

def plot_ablation_bar(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values('f0.5', ascending=True)
    colors    = ['#4C72B0' if idx in VARIANT_ORDER else '#DD8452' for idx in df_sorted.index]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(df_sorted) + 1.5))
    bars = ax.barh(
        [VARIANT_LABELS.get(v, v) for v in df_sorted.index],
        df_sorted['f0.5'].values,
        color=colors, alpha=0.85,
    )
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_xlabel('ERRANT F0.5')
    ax.set_title('Ablation leaderboard')
    ax.set_xlim(0, df_sorted['f0.5'].max() * 1.15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    fig.tight_layout()
    out = RESULTS / 'pe_ablation_bar.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


# ── Plot 4: CoLA before/after T5 ──────────────────────────────────────────────

def plot_cola_improvement(df: pd.DataFrame) -> None:
    if 'cola_B' not in df.columns or 'cola_C' not in df.columns:
        return
    x      = np.arange(len(df))
    width  = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, df['cola_B'].values, width, label='Before T5 (B)', color='#C44E52', alpha=0.85)
    ax.bar(x + width/2, df['cola_C'].values, width, label='After T5 (C)',  color='#55A868', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in df.index], fontsize=10)
    ax.set_ylabel('CoLA acceptability rate')
    ax.set_title('T5 correction effect on CoLA score (B → C)')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    out = RESULTS / 'pe_cola_improvement.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved → {out}')


def main() -> None:
    df = load_leaderboard()
    plot_metrics_bar(df)
    plot_metrics_heatmap(df)
    plot_cola_improvement(df)

    if ABLATION.exists():
        abl_df = pd.read_csv(ABLATION, index_col=0)
        plot_ablation_bar(abl_df)
    else:
        print(f'No ablation file at {ABLATION} — skipping ablation plot.')


if __name__ == '__main__':
    main()
