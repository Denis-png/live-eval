"""
evaluation.py — T5 correction + CoLA + bigram Jaccard + ERRANT per dataset tier.

Run from Denis/datasets_experiments/:
    python evaluation.py

Input:  data/generated/ds_{name}_generated.csv
Output:
    data/corrected/ds_{name}_corrected.csv  (adds t5_corrected column)
    data/results/datasets_comparison.csv
"""
import warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
import spacy, errant

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent

(ROOT / 'data/corrected').mkdir(parents=True, exist_ok=True)
(ROOT / 'data/results').mkdir(parents=True, exist_ok=True)

DATASETS = ['conll14', 'fce', 'c4_200m']
TIER_LABELS = {
    'conll14': 'Test set',
    'fce':     'Benchmark',
    'c4_200m': 'General',
}

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
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc='CoLA', leave=False):
        preds = cola_scorer(texts[i:i+batch_size], truncation=True, max_length=128)
        scores.extend(1.0 if p['label'] == 'LABEL_1' else 0.0 for p in preds)
    return scores


# ── Bigram Jaccard ─────────────────────────────────────────────────────────────

def bigram_jaccard(s1: str, s2: str) -> float:
    def bigrams(s):
        toks = s.lower().split()
        return set(zip(toks, toks[1:])) if len(toks) > 1 else set()
    b1, b2 = bigrams(s1), bigrams(s2)
    if not b1 and not b2:
        return 0.0
    return len(b1 & b2) / len(b1 | b2)


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


# ── T5 correction (with checkpoint) ───────────────────────────────────────────

corrected: dict[str, pd.DataFrame] = {}
for name in DATASETS:
    gen_path = ROOT / f'data/generated/ds_{name}_generated.csv'
    cor_path = ROOT / f'data/corrected/ds_{name}_corrected.csv'
    if not gen_path.exists():
        print(f'Missing: {gen_path.name} — run generation.py first.')
        continue
    if cor_path.exists():
        print(f'Checkpoint: {cor_path.name} — loading.')
        corrected[name] = pd.read_csv(cor_path)
        continue
    df = pd.read_csv(gen_path)
    df['generated'] = df['generated'].fillna('[GENERATION FAILED]')
    print(f'\nT5 correction: {name}')
    t5_out = correct_sentences(df['generated'].tolist())
    df_out = pd.DataFrame({
        'original':     df['original'],
        'ground_truth': df['ground_truth'],
        'tier':         df['tier'],
        'generated':    df['generated'],
        't5_corrected': t5_out,
    })
    df_out.to_csv(cor_path, index=False)
    corrected[name] = df_out
    print(f'  Saved → {cor_path}')

if not corrected:
    raise SystemExit('No corrected datasets found — run generation.py first.')

# ── Per-dataset metrics ────────────────────────────────────────────────────────

rows = []
for name, df in corrected.items():
    print(f'\nEvaluating: {name}')

    a_sc  = cola_scores(df['original'].tolist())
    b_sc  = cola_scores(df['generated'].tolist())
    c_sc  = cola_scores(df['t5_corrected'].tolist())

    jaccards = [
        bigram_jaccard(o, g)
        for o, g in zip(df['original'], df['generated'])
    ]
    corr_jaccards = [
        bigram_jaccard(g, t)
        for g, t in zip(df['generated'], df['t5_corrected'])
    ]

    ref_edits = [
        get_edits(o, g)
        for o, g in tqdm(
            zip(df['original'].tolist(), df['ground_truth'].tolist()),
            desc='  ref ERRANT', total=len(df), leave=False,
        )
    ]
    hyp_edits = [
        get_edits(g, t)
        for g, t in tqdm(
            zip(df['generated'].tolist(), df['t5_corrected'].tolist()),
            desc='  hyp ERRANT', total=len(df), leave=False,
        )
    ]

    rows.append({
        'dataset':       name,
        'tier':          TIER_LABELS[name],
        'errant_f0.5':   round(dist_f05(edit_type_dist(hyp_edits), edit_type_dist(ref_edits)), 3),
        'cola_A':        round(np.mean(a_sc), 3),
        'cola_B':        round(np.mean(b_sc), 3),
        'cola_C':        round(np.mean(c_sc), 3),
        'bigram_jaccard':  round(np.mean(jaccards), 3),
        'pct_memorised':   round(np.mean([s > 0.3 for s in jaccards]), 3),
        'corr_extent':     round(1 - np.mean(corr_jaccards), 3),
        'B<A':           bool(np.mean(b_sc) < np.mean(a_sc)),
        'C>B':           bool(np.mean(c_sc) > np.mean(b_sc)),
    })

comparison = pd.DataFrame(rows).set_index('dataset')
out_path   = ROOT / 'data/results/datasets_comparison.csv'
comparison.to_csv(out_path)
print('\n=== DATASET TIER COMPARISON ===')
print(comparison.to_string())
print(f'\nSaved → {out_path}')
