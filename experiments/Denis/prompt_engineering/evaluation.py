"""
evaluation.py — T5 correction + CoLA + semantic similarity + ERRANT for all variants.

Run from Denis/prompt_engineering/:
    python evaluation.py

Input:  data/generated/pe_{variant}.csv
Output:
    data/corrected/pe_{variant}_corrected.csv  (adds t5_corrected column)
    data/results/prompt_leaderboard.csv
"""
import os, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import spacy, errant

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent

(ROOT / 'data/corrected').mkdir(parents=True, exist_ok=True)
(ROOT / 'data/results').mkdir(parents=True, exist_ok=True)

VARIANTS = ['zero_shot', 'one_shot', 'few_shot_3', 'few_shot_12', 'cot']

# ── T5 GEC ─────────────────────────────────────────────────────────────────────

T5_MODEL = 'vennify/t5-base-grammar-correction'
t5_tok = AutoTokenizer.from_pretrained(T5_MODEL)
t5_mdl = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL)
t5_mdl.eval()
print('T5 loaded')


def correct_sentences(sentences: list[str], batch_size: int = 8,
                      prefix: str = 'grammar: ') -> list[str]:
    results = []
    for i in tqdm(range(0, len(sentences), batch_size), desc='T5 GEC', leave=False):
        batch  = [prefix + s for s in sentences[i:i+batch_size]]
        inputs = t5_tok(batch, return_tensors='pt', padding=True,
                        truncation=True, max_length=128)
        with torch.no_grad():
            outputs = t5_mdl.generate(**inputs, max_new_tokens=128)
        results.extend(t5_tok.batch_decode(outputs, skip_special_tokens=True))
    return results


# ── CoLA ───────────────────────────────────────────────────────────────────────

device = 0 if torch.cuda.is_available() else -1
cola_scorer = hf_pipeline(
    'text-classification',
    model='textattack/bert-base-uncased-CoLA',
    device=device,
)
print('CoLA scorer loaded')


def cola_scores(texts: list[str], batch_size: int = 32) -> list[float]:
    """Return 1.0 (grammatically acceptable) or 0.0 per sentence."""
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc='CoLA', leave=False):
        preds = cola_scorer(texts[i:i+batch_size], truncation=True, max_length=128)
        scores.extend(1.0 if p['label'] == 'LABEL_1' else 0.0 for p in preds)
    return scores


# ── Semantic similarity ────────────────────────────────────────────────────────

def tfidf_sim(texts_a: list[str], texts_b: list[str]) -> list[float]:
    """Pairwise TF-IDF cosine between two aligned lists."""
    all_texts = texts_a + texts_b
    tfidf = TfidfVectorizer().fit_transform(all_texts)
    n = len(texts_a)
    return [float(cos_sim(tfidf[i], tfidf[n + i])[0, 0]) for i in range(n)]


# ── ERRANT ─────────────────────────────────────────────────────────────────────

try:
    spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)

annotator = errant.load('en')


def get_edits(src_sent: str, tgt_sent: str):
    return annotator.annotate(annotator.parse(src_sent), annotator.parse(tgt_sent))


def edit_type_dist(edit_lists) -> dict[str, float]:
    c = Counter(e.type for edits in edit_lists for e in edits)
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def dist_f05(hyp: dict, ref: dict) -> float:
    types = set(hyp) | set(ref)
    tp = sum(min(hyp.get(t, 0), ref.get(t, 0)) for t in types)
    fp = sum(max(hyp.get(t, 0) - ref.get(t, 0), 0) for t in types)
    fn = sum(max(ref.get(t, 0) - hyp.get(t, 0), 0) for t in types)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return (1.25 * p * r) / (0.25 * p + r) if 0.25 * p + r > 0 else 0


# ── T5 correction pass (with checkpoint) ──────────────────────────────────────

corrected: dict[str, pd.DataFrame] = {}
for variant in VARIANTS:
    gen_path = ROOT / f'data/generated/pe_{variant}.csv'
    cor_path = ROOT / f'data/corrected/pe_{variant}_corrected.csv'
    if not gen_path.exists():
        print(f'Missing generated file: {gen_path.name} — run generation.py first.')
        continue
    if cor_path.exists():
        print(f'Checkpoint: {cor_path.name} — loading.')
        corrected[variant] = pd.read_csv(cor_path)
        continue
    df_v = pd.read_csv(gen_path)
    df_v['generated'] = df_v['generated'].fillna('[GENERATION FAILED]')
    print(f'\nT5 correction: {variant}')
    t5_out = correct_sentences(df_v['generated'].tolist())
    # Column order: original, ground_truth, generated, t5_corrected
    df_v = df_v[['original', 'ground_truth', 'generated']].copy()
    df_v['t5_corrected'] = t5_out
    df_v.to_csv(cor_path, index=False)
    corrected[variant] = df_v
    print(f'  Saved → {cor_path}')

if not corrected:
    raise SystemExit('No corrected variants found — run generation.py first.')

# ── Compute reference ERRANT distribution (first available variant) ────────────

first_df   = next(iter(corrected.values()))
ref_edits  = [
    get_edits(o, g)
    for o, g in tqdm(
        zip(first_df['original'].tolist(), first_df['ground_truth'].tolist()),
        desc='Ref ERRANT', total=len(first_df),
    )
]
ref_dist = edit_type_dist(ref_edits)
print('\nReference error-type distribution (top 10):')
for k, v in sorted(ref_dist.items(), key=lambda x: -x[1])[:10]:
    print(f'  {k}: {v:.3f}')

# ── Per-variant metrics ────────────────────────────────────────────────────────

rows = []
for variant, df_v in corrected.items():
    print(f'\nEvaluating: {variant}')
    a_sc = cola_scores(df_v['original'].tolist())
    b_sc = cola_scores(df_v['generated'].tolist())
    c_sc = cola_scores(df_v['t5_corrected'].tolist())
    sims = tfidf_sim(df_v['original'].tolist(), df_v['generated'].tolist())
    corr_sims = tfidf_sim(df_v['generated'].tolist(), df_v['t5_corrected'].tolist())
    hyp_edits = [
        get_edits(g, t)
        for g, t in tqdm(
            zip(df_v['generated'].tolist(), df_v['t5_corrected'].tolist()),
            desc=f'  ERRANT {variant}', total=len(df_v), leave=False,
        )
    ]
    hyp_dist = edit_type_dist(hyp_edits)
    rows.append({
        'variant': variant,
        'f0.5':    round(dist_f05(hyp_dist, ref_dist), 3),
        'n_edits': round(np.mean([len(e) for e in hyp_edits]), 2),
        'cola_A':  round(np.mean(a_sc), 3),
        'cola_B':  round(np.mean(b_sc), 3),
        'cola_C':  round(np.mean(c_sc), 3),
        'sem_sim':     round(np.mean(sims), 3),
        'corr_extent': round(1 - np.mean(corr_sims), 3),
        'B<A':         bool(np.mean(b_sc) < np.mean(a_sc)),
        'C>B':     bool(np.mean(c_sc) > np.mean(b_sc)),
    })

leaderboard = (
    pd.DataFrame(rows)
    .set_index('variant')
    .sort_values('f0.5', ascending=False)
)

out_path = ROOT / 'data/results/prompt_leaderboard.csv'
leaderboard.to_csv(out_path)
print('\n=== PROMPT LEADERBOARD (sorted by ERRANT F0.5) ===')
print(leaderboard.to_string())
print(f'\nSaved → {out_path}')
