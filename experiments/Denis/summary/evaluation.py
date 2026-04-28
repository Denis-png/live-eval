"""
evaluation.py — Multi-model GEC correction + metrics on original FCE vs generated benchmark.

Run from Denis/:
    python summary/evaluation.py

Input:  data/summary/summary_generated.csv

Output: data/summary/summary_corrected.csv          (generated sentences + all model corrections)
        data/summary/summary_original_corrected.csv (original FCE sentences + all model corrections)
        data/summary/summary_comparison.csv          (aggregated metrics: model × dataset)

GEC models:
  t5_base      — vennify/t5-base-grammar-correction
  coedit_large — grammarly/coedit-large
  prithivida   — prithivida/grammar_error_correcter_v1
  haiku        — claude-haiku-4-5 (zero-shot via Anthropic API)

Metrics:
  errant_f0.5    — distribution-level F0.5 vs FCE reference (cross-dataset comparison)
  errant_f0.5_gt — sentence-level F0.5 using per-sentence ground truth (actual correction accuracy)
  CoLA (input/corrected), correction_extent, n_edits
"""
import os, time, re, warnings, math
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
import anthropic
import spacy, errant

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent

load_dotenv(ROOT / '.env')
if not os.getenv('ANTHROPIC_API_KEY'):
    load_dotenv(ROOT.parent / 'Denis/.env')
assert os.getenv('ANTHROPIC_API_KEY'), 'ANTHROPIC_API_KEY not found'

(ROOT / 'data/summary').mkdir(parents=True, exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else -1
HF_TOKEN = os.getenv('HF_TOKEN')

# ── Load data ──────────────────────────────────────────────────────────────────

def load_m2(path: str, annotator_id: int = 0) -> pd.DataFrame:
    records = []
    with open(path, encoding='utf-8') as f:
        blocks = f.read().strip().split('\n\n')
    for block in blocks:
        lines = [l for l in block.strip().splitlines() if l.strip()]
        if not lines or not lines[0].startswith('S '):
            continue
        src_tokens = lines[0][2:].split()
        edits = []
        for line in lines[1:]:
            if not line.startswith('A '):
                continue
            parts = line[2:].split('|||')
            if len(parts) < 6:
                continue
            span = parts[0].split()
            start, end = int(span[0]), int(span[1])
            correction = parts[2]
            ann = int(parts[5])
            if ann == annotator_id and correction != '-NONE-':
                edits.append((start, end, correction))
        if not edits:
            continue
        tgt = src_tokens[:]
        for start, end, correction in sorted(edits, key=lambda x: -x[0]):
            tgt[start:end] = correction.split() if correction else []
        records.append({'original': ' '.join(src_tokens), 'ground_truth': ' '.join(tgt)})
    return pd.DataFrame(records)


gen_path = ROOT / 'data/summary/summary_generated.csv'
if not gen_path.exists():
    raise SystemExit('summary_generated.csv not found — run generation.py first.')

df_gen = pd.read_csv(gen_path)
df_gen['generated'] = df_gen['generated'].fillna('[GENERATION FAILED]')

# Backward-compat: old CSVs used 'ground_truth' instead of 'fce_ground_truth'
if 'fce_ground_truth' not in df_gen.columns and 'ground_truth' in df_gen.columns:
    df_gen = df_gen.rename(columns={'ground_truth': 'fce_ground_truth'})
if 'gen_ground_truth' not in df_gen.columns:
    df_gen['gen_ground_truth'] = ''
df_gen['fce_ground_truth'] = df_gen['fce_ground_truth'].fillna('')
df_gen['gen_ground_truth']  = df_gen['gen_ground_truth'].fillna('')

print(f'Loaded {len(df_gen)} generated sentences')

# ── GEC model definitions ──────────────────────────────────────────────────────

HF_MODELS = {
    't5_base': {
        'model_id': 'vennify/t5-base-grammar-correction',
        'prefix':   'grammar: ',
        'token':    None,
    },
    'coedit_large': {
        'model_id': 'grammarly/coedit-large',
        'prefix':   'Fix the grammar of this sentence: ',
        'token':    None,
    },
    'prithivida': {
        # Trained with HappyTransformer; loads fine as standard seq2seq.
        # Uses no special prefix — input is the raw sentence.
        'model_id': 'prithivida/grammar_error_correcter_v1',
        'prefix':   '',
        'token':    HF_TOKEN,
    },
}

# ── HF correction utilities ────────────────────────────────────────────────────

def load_hf_model(cfg: dict):
    tok = AutoTokenizer.from_pretrained(cfg['model_id'], token=cfg['token'])
    mdl = AutoModelForSeq2SeqLM.from_pretrained(cfg['model_id'], token=cfg['token'])
    mdl.eval()
    if torch.cuda.is_available():
        mdl = mdl.cuda()
    return tok, mdl


def correct_hf(sentences: list[str], tok, mdl, prefix: str,
               batch_size: int = 8) -> list[str]:
    results = []
    for i in tqdm(range(0, len(sentences), batch_size), desc='  HF GEC', leave=False):
        batch  = [prefix + s for s in sentences[i:i+batch_size]]
        inputs = tok(batch, return_tensors='pt', padding=True,
                     truncation=True, max_length=128)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mdl.generate(**inputs, max_new_tokens=128)
        results.extend(tok.batch_decode(outputs, skip_special_tokens=True))
    return results


def unload_hf(tok, mdl):
    del tok, mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Haiku (API) correction ─────────────────────────────────────────────────────

api_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

HAIKU_SYSTEM = (
    'You are a grammatical error corrector. '
    'For each numbered sentence, correct all grammatical errors and return only the '
    'corrected version with the same number. Make minimal changes — fix grammar only, '
    'preserve the original meaning and topic. '
    'Return only the numbered corrected sentences, one per line.'
)


def make_batch_input(sentences: list[str]) -> str:
    return '\n'.join(f'{i+1}. {s}' for i, s in enumerate(sentences))


def parse_batch_output(text: str, expected: int) -> list[str]:
    lines = [
        re.sub(r'^[0-9]+[.)\s]+', '', l).strip()
        for l in text.strip().splitlines()
        if re.match(r'^[0-9]+[.)]', l.strip())
    ]
    if len(lines) == expected:
        return lines
    fallback = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(fallback) == expected:
        return fallback
    raise ValueError(f'Expected {expected}, got {len(lines)}/{len(fallback)}:\n{text}')


def correct_haiku(sentences: list[str], batch_size: int = 5,
                  sleep: int = 10) -> list[str]:
    results = []
    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
    failed_batches = []

    for idx, batch in enumerate(tqdm(batches, desc='  Haiku GEC')):
        success = False
        for attempt in range(7):
            try:
                response = api_client.messages.create(
                    model='claude-haiku-4-5',
                    max_tokens=1024,
                    system=[{
                        'type': 'text',
                        'text': HAIKU_SYSTEM,
                        'cache_control': {'type': 'ephemeral'},
                    }],
                    messages=[{'role': 'user', 'content': make_batch_input(batch)}],
                )
                corrected = parse_batch_output(response.content[0].text, len(batch))
                results.extend(corrected)
                success = True
                break
            except ValueError:
                time.sleep(10 * (2 ** attempt))
            except anthropic.RateLimitError:
                time.sleep(60 * (2 ** attempt))
        if not success:
            results.extend(['[CORRECTION FAILED]'] * len(batch))
            failed_batches.append(idx)
        time.sleep(sleep)

    if failed_batches:
        print(f'  Warning: {len(failed_batches)} Haiku batch(es) failed')
    return results


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def ckpt_path(model_name: str, dataset: str) -> Path:
    return ROOT / f'data/summary/ckpt_{model_name}_{dataset}.csv'


def load_or_correct_hf(model_name: str, cfg: dict,
                        sentences_gen: list[str],
                        sentences_orig: list[str]) -> tuple[list[str], list[str]]:
    gen_ckpt  = ckpt_path(model_name, 'generated')
    orig_ckpt = ckpt_path(model_name, 'original')

    if gen_ckpt.exists() and orig_ckpt.exists():
        print(f'  Checkpoint found for {model_name} — loading.')
        return (
            pd.read_csv(gen_ckpt)['corrected'].tolist(),
            pd.read_csv(orig_ckpt)['corrected'].tolist(),
        )

    print(f'  Loading {cfg["model_id"]}...')
    tok, mdl = load_hf_model(cfg)
    gen_corr  = correct_hf(sentences_gen,  tok, mdl, cfg['prefix'])
    orig_corr = correct_hf(sentences_orig, tok, mdl, cfg['prefix'])
    unload_hf(tok, mdl)

    pd.DataFrame({'corrected': gen_corr}).to_csv(gen_ckpt, index=False)
    pd.DataFrame({'corrected': orig_corr}).to_csv(orig_ckpt, index=False)
    return gen_corr, orig_corr


def load_or_correct_haiku(sentences_gen: list[str],
                           sentences_orig: list[str]) -> tuple[list[str], list[str]]:
    gen_ckpt  = ckpt_path('haiku', 'generated')
    orig_ckpt = ckpt_path('haiku', 'original')

    if gen_ckpt.exists() and orig_ckpt.exists():
        print('  Checkpoint found for haiku — loading.')
        return (
            pd.read_csv(gen_ckpt)['corrected'].tolist(),
            pd.read_csv(orig_ckpt)['corrected'].tolist(),
        )

    print('  Correcting generated sentences...')
    gen_corr  = correct_haiku(sentences_gen)
    print('  Correcting original sentences...')
    orig_corr = correct_haiku(sentences_orig)

    pd.DataFrame({'corrected': gen_corr}).to_csv(gen_ckpt, index=False)
    pd.DataFrame({'corrected': orig_corr}).to_csv(orig_ckpt, index=False)
    return gen_corr, orig_corr


# ── Run all models ─────────────────────────────────────────────────────────────

sentences_gen  = df_gen['generated'].tolist()
sentences_orig = df_gen['original'].tolist()
fce_gt_list    = df_gen['fce_ground_truth'].tolist()
gen_gt_list    = df_gen['gen_ground_truth'].tolist()

corrections_gen:  dict[str, list[str]] = {}
corrections_orig: dict[str, list[str]] = {}

for model_name, cfg in HF_MODELS.items():
    print(f'\n[{model_name}]')
    g, o = load_or_correct_hf(model_name, cfg, sentences_gen, sentences_orig)
    corrections_gen[model_name]  = g
    corrections_orig[model_name] = o

print('\n[haiku]')
g, o = load_or_correct_haiku(sentences_gen, sentences_orig)
corrections_gen['haiku']  = g
corrections_orig['haiku'] = o

# ── Save corrected CSVs ────────────────────────────────────────────────────────

df_corr_gen = df_gen[['original', 'fce_ground_truth', 'generated', 'gen_ground_truth']].copy()
for m, corr in corrections_gen.items():
    df_corr_gen[f'{m}_corrected'] = corr
df_corr_gen.to_csv(ROOT / 'data/summary/summary_corrected.csv', index=False)
print(f'\nSaved → data/summary/summary_corrected.csv')

df_corr_orig = df_gen[['original', 'fce_ground_truth']].copy()
for m, corr in corrections_orig.items():
    df_corr_orig[f'{m}_corrected'] = corr
df_corr_orig.to_csv(ROOT / 'data/summary/summary_original_corrected.csv', index=False)
print(f'Saved → data/summary/summary_original_corrected.csv')

# ── Metrics ────────────────────────────────────────────────────────────────────

# CoLA scorer
cola_scorer = hf_pipeline(
    'text-classification',
    model='textattack/bert-base-uncased-CoLA',
    device=DEVICE,
)


def cola_scores(texts: list[str], batch_size: int = 32) -> list[float]:
    scores = []
    for i in range(0, len(texts), batch_size):
        preds = cola_scorer(texts[i:i+batch_size], truncation=True, max_length=128)
        scores.extend(1.0 if p['label'] == 'LABEL_1' else 0.0 for p in preds)
    return scores


# ERRANT
try:
    spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)

annotator = errant.load('en')


def get_edits(src: str, tgt: str):
    return annotator.annotate(annotator.parse(src), annotator.parse(tgt))


def edit_type_dist(edit_lists) -> dict:
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


def errant_f05_gt(sources: list[str], hypotheses: list[str],
                  references: list[str]) -> float:
    """Sentence-level ERRANT F0.5 using per-sentence ground-truth references.

    Skips rows where the reference is empty or a failure placeholder.
    """
    tp = fp = fn = 0
    _skip = {'', '[GENERATION FAILED]', '[CORRECTION FAILED]'}
    for src, hyp, ref in zip(sources, hypotheses, references):
        if ref in _skip:
            continue
        src_p = annotator.parse(src)
        hyp_c = Counter(e.type for e in annotator.annotate(src_p, annotator.parse(hyp)))
        ref_c = Counter(e.type for e in annotator.annotate(src_p, annotator.parse(ref)))
        for t in set(hyp_c) | set(ref_c):
            h, r = hyp_c.get(t, 0), ref_c.get(t, 0)
            tp += min(h, r)
            fp += max(h - r, 0)
            fn += max(r - h, 0)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return (1.25 * p * r) / (0.25 * p + r) if 0.25 * p + r > 0 else 0


def bigram_jaccard(s1: str, s2: str) -> float:
    def bigrams(s):
        toks = s.lower().split()
        return set(zip(toks, toks[1:])) if len(toks) > 1 else set()
    b1, b2 = bigrams(s1), bigrams(s2)
    if not b1 and not b2:
        return 0.0
    return len(b1 & b2) / len(b1 | b2)


def _ngram_counts(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def gleu_sentence(hypothesis: str, reference: str, max_n: int = 4) -> float:
    """Sentence-level GEC GLEU (Napoles et al. 2015).

    GLEU_n = |hyp ∩ ref| / max(|hyp_n|, |ref_n|); penalises both over- and
    under-generation. Final score = geometric mean across n=1..max_n.
    """
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    if not hyp or not ref:
        return 0.0
    log_score = 0.0
    for n in range(1, max_n + 1):
        hyp_ng = _ngram_counts(hyp, n)
        ref_ng = _ngram_counts(ref, n)
        matches = sum(min(hyp_ng[g], ref_ng[g]) for g in hyp_ng)
        total   = max(sum(hyp_ng.values()), sum(ref_ng.values()))
        if total == 0:
            continue
        if matches == 0:
            return 0.0
        log_score += math.log(matches / total)
    return math.exp(log_score / max_n)


def mean_gleu(hypotheses: list[str], references: list[str]) -> float:
    """Mean sentence GLEU, skipping empty/failed references."""
    _skip = {'', '[GENERATION FAILED]', '[CORRECTION FAILED]'}
    scores = [
        gleu_sentence(h, r)
        for h, r in zip(hypotheses, references)
        if r not in _skip
    ]
    return float(np.mean(scores)) if scores else 0.0


# Reference ERRANT distribution: original FCE errors (original → fce_ground_truth)
print('\nComputing FCE reference ERRANT distribution...')
ref_edits = [
    get_edits(o, g)
    for o, g in tqdm(
        zip(sentences_orig, fce_gt_list),
        desc='Ref ERRANT', total=len(df_gen),
    )
]
ref_dist = edit_type_dist(ref_edits)
print('Top error types:', sorted(ref_dist.items(), key=lambda x: -x[1])[:5])

# CoLA of inputs (computed once each)
print('\nCoLA: original FCE sentences...')
cola_orig_input = cola_scores(sentences_orig)
print('CoLA: generated sentences...')
cola_gen_input  = cola_scores(sentences_gen)

# ── Per-model metrics ──────────────────────────────────────────────────────────

rows = []
all_model_names = list(HF_MODELS.keys()) + ['haiku']

for model_name in all_model_names:
    for dataset, input_sents, corr_sents, cola_input, gt_sents in [
        ('generated', sentences_gen,  corrections_gen[model_name],  cola_gen_input,  gen_gt_list),
        ('original',  sentences_orig, corrections_orig[model_name], cola_orig_input, fce_gt_list),
    ]:
        print(f'\nMetrics: {model_name} × {dataset}')
        corr_sents = [s if isinstance(s, str) else '[CORRECTION FAILED]' for s in corr_sents]

        cola_corr = cola_scores(corr_sents)

        hyp_edits = [
            get_edits(inp, corr)
            for inp, corr in tqdm(
                zip(input_sents, corr_sents),
                desc=f'  ERRANT {model_name}/{dataset}', total=len(input_sents), leave=False,
            )
        ]
        hyp_dist = edit_type_dist(hyp_edits)

        jac = [bigram_jaccard(inp, corr) for inp, corr in zip(input_sents, corr_sents)]

        f05_gt   = errant_f05_gt(input_sents, corr_sents, gt_sents)
        gleu_val = mean_gleu(corr_sents, gt_sents)

        rows.append({
            'model':             model_name,
            'dataset':           dataset,
            'errant_f0.5':       round(dist_f05(hyp_dist, ref_dist), 3),
            'errant_f0.5_gt':    round(f05_gt, 3),
            'gleu_gt':           round(gleu_val, 3),
            'cola_input':        round(np.mean(cola_input), 3),
            'cola_corrected':    round(np.mean(cola_corr), 3),
            'cola_delta':        round(np.mean(cola_corr) - np.mean(cola_input), 3),
            'correction_extent': round(1 - np.mean(jac), 3),
            'n_edits':           round(np.mean([len(e) for e in hyp_edits]), 2),
        })
        print(f'  F0.5={rows[-1]["errant_f0.5"]}  F0.5_gt={rows[-1]["errant_f0.5_gt"]}  '
              f'GLEU={rows[-1]["gleu_gt"]}  CoLA Δ={rows[-1]["cola_delta"]:+.3f}  '
              f'corr_extent={rows[-1]["correction_extent"]}')

# ── Save comparison ────────────────────────────────────────────────────────────

comparison = pd.DataFrame(rows)
out_path   = ROOT / 'data/summary/summary_comparison.csv'
comparison.to_csv(out_path, index=False)
print('\n=== SUMMARY COMPARISON ===')
print(comparison.to_string(index=False))
print(f'\nSaved → {out_path}')

# Clean up per-model checkpoint files
for m in all_model_names:
    for ds in ('generated', 'original'):
        p = ckpt_path(m, ds)
        if p.exists():
            p.unlink()
print('Checkpoints cleaned up.')
