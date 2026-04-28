"""
ablation.py — Ablation study: instruction wording × temperature × n_examples.

Run from Denis/prompt_engineering/ (after evaluation.py):
    python ablation.py

Output: data/results/ablation_leaderboard.csv

Variables swept (one at a time vs few_shot_12 baseline at T=0.7):
  - Instruction wording: concise vs detailed
  - Temperature: 0.3 / 1.0  (0.7 is the main few_shot_12 result)
  - n_examples: 0 / 1 / 3 / 12  (from main run slices)
  - CoT: off / on              (from main run slices)
"""
import os, json, time, re, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from dotenv import load_dotenv
import anthropic
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import spacy, errant

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent

load_dotenv(ROOT / '.env')
if not os.getenv('ANTHROPIC_API_KEY'):
    load_dotenv(ROOT.parent / 'Denis/.env')
assert os.getenv('ANTHROPIC_API_KEY'), 'ANTHROPIC_API_KEY not found'

(ROOT / 'data/generated').mkdir(parents=True, exist_ok=True)
(ROOT / 'data/results').mkdir(parents=True, exist_ok=True)

N_ABL = 30   # sentences per ablation config

# ── Load sample + reference ────────────────────────────────────────────────────

df_sample = pd.read_csv(ROOT / 'data/sampled/pe_sample.csv')
abl_src   = df_sample['original'].tolist()[:N_ABL]
abl_gt    = df_sample['ground_truth'].tolist()[:N_ABL]

with open(ROOT / 'data/gec_error_types.json') as f:
    error_types = json.load(f)['error_types']
with open(ROOT / 'data/gec_few_shot_examples.json') as f:
    few_shot_data = json.load(f)
demonstrations = few_shot_data['demonstrations']

# ── LLM client ─────────────────────────────────────────────────────────────────

MODEL  = 'claude-haiku-4-5'
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))


def call_llm(system: str, static_user: str, dynamic_user: str,
             temperature: float = 0.7) -> str:
    user_content: list[dict] = []
    if static_user:
        user_content.append({
            'type': 'text',
            'text': static_user,
            'cache_control': {'type': 'ephemeral'},
        })
    user_content.append({'type': 'text', 'text': dynamic_user})
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        temperature=temperature,
        system=[{
            'type': 'text',
            'text': system,
            'cache_control': {'type': 'ephemeral'},
        }],
        messages=[{'role': 'user', 'content': user_content}],
    )
    return response.content[0].text


# ── Batch utilities (local copies to keep this script self-contained) ──────────

def make_batch_input(sentences):
    return '\n'.join(f'{i+1}. {s}' for i, s in enumerate(sentences))


def parse_batch_output(text, expected):
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
    raise ValueError(f'Expected {expected} lines, got {len(lines)}/{len(fallback)}:\n{text}')


def invoke_with_retry(system, static_user, sentences, temperature=0.7,
                      max_retries=7, base_delay=10):
    dynamic_user = make_batch_input(sentences)
    last_error   = None
    for attempt in range(max_retries):
        try:
            result = call_llm(system, static_user, dynamic_user, temperature)
            return parse_batch_output(result, len(sentences))
        except ValueError as e:
            last_error = e
            time.sleep(base_delay * (2 ** attempt))
        except anthropic.RateLimitError:
            time.sleep(60 * (2 ** attempt))
    raise RuntimeError(f'Failed after {max_retries} retries. Last: {last_error}')


def run_generation(system, static_user, sentences, temperature=0.7,
                   batch_size=3, sleep=15, desc='ABL'):
    results = []
    for batch in tqdm(
        [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)],
        desc=desc,
    ):
        results.extend(invoke_with_retry(system, static_user, batch, temperature))
        time.sleep(sleep)
    return results


# ── Prompt builders ────────────────────────────────────────────────────────────

FS_BASE_SYSTEM = (
    'You are a grammatical error generator. '
    'For each sentence below, write a completely NEW sentence on a different topic '
    'that contains the same type of grammatical error as the input. '
    'The generated sentence MUST contain a grammatical error — do NOT produce a correct sentence. '
    'Do NOT correct the given sentence.\n'
    'Return only the generated sentences, one per line, preserving the numbering.'
)

INSTR_CONCISE = (
    'Generate a new sentence with the same grammatical error type. '
    'The generated sentence MUST contain an error. '
    'New sentence only, numbered.'
)
INSTR_DETAILED = (
    'You are an expert linguist specializing in grammatical error generation. '
    'Create a completely new, original sentence that exhibits the same category of '
    'grammatical error as the given sentence. Use an entirely different topic. '
    'The generated sentence MUST contain a grammatical error. '
    'Preserve the original numbering. Output only the generated sentences.'
)


def make_few_shot_block(pairs):
    lines = []
    for i, (inp, out) in enumerate(pairs, 1):
        lines += [f'Example {i}:', f'Input:  {inp}', f'Output: {out}', '']
    return '\n'.join(lines) + 'Now generate for:\n'


pairs_12 = [
    (demonstrations[et['name']]['input'], demonstrations[et['name']]['output'])
    for et in error_types
]

FS12_STATIC = make_few_shot_block(pairs_12)

# Each entry: (system, static_user, temperature)
NEW_CONFIGS = {
    'instr_concise':  (INSTR_CONCISE,  FS12_STATIC, 0.7),
    'instr_detailed': (INSTR_DETAILED, FS12_STATIC, 0.7),
    'temp_0.3':       (FS_BASE_SYSTEM, FS12_STATIC, 0.3),
    'temp_1.0':       (FS_BASE_SYSTEM, FS12_STATIC, 1.0),
}

# ── T5 + metrics ───────────────────────────────────────────────────────────────

T5_MODEL = 'vennify/t5-base-grammar-correction'
t5_tok = AutoTokenizer.from_pretrained(T5_MODEL)
t5_mdl = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL)
t5_mdl.eval()

device = 0 if torch.cuda.is_available() else -1
cola_scorer = hf_pipeline(
    'text-classification',
    model='textattack/bert-base-uncased-CoLA',
    device=device,
)

try:
    spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
annotator = errant.load('en')


def correct_sentences(sentences, batch_size=8, prefix='grammar: '):
    results = []
    for i in range(0, len(sentences), batch_size):
        batch  = [prefix + s for s in sentences[i:i+batch_size]]
        inputs = t5_tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = t5_mdl.generate(**inputs, max_new_tokens=128)
        results.extend(t5_tok.batch_decode(outputs, skip_special_tokens=True))
    return results


def cola_scores(texts, batch_size=32):
    scores = []
    for i in range(0, len(texts), batch_size):
        preds = cola_scorer(texts[i:i+batch_size], truncation=True, max_length=128)
        scores.extend(1.0 if p['label'] == 'LABEL_1' else 0.0 for p in preds)
    return scores


def tfidf_sim(texts_a, texts_b):
    tfidf = TfidfVectorizer().fit_transform(texts_a + texts_b)
    n = len(texts_a)
    return [float(cos_sim(tfidf[i], tfidf[n + i])[0, 0]) for i in range(n)]


def get_edits(src_sent, tgt_sent):
    return annotator.annotate(annotator.parse(src_sent), annotator.parse(tgt_sent))


def edit_type_dist(edit_lists):
    c = Counter(e.type for edits in edit_lists for e in edits)
    total = sum(c.values()) or 1
    return {k: v / total for k, v in c.items()}


def dist_f05(hyp, ref):
    types = set(hyp) | set(ref)
    tp = sum(min(hyp.get(t, 0), ref.get(t, 0)) for t in types)
    fp = sum(max(hyp.get(t, 0) - ref.get(t, 0), 0) for t in types)
    fn = sum(max(ref.get(t, 0) - hyp.get(t, 0), 0) for t in types)
    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    return (1.25 * p * r) / (0.25 * p + r) if 0.25 * p + r > 0 else 0


ref_edits = [get_edits(o, g) for o, g in zip(abl_src, abl_gt)]
ref_dist  = edit_type_dist(ref_edits)


def eval_generated(name, generated, orig_list, gt_list):
    t5_c = correct_sentences(generated)
    b_sc = cola_scores(generated)
    a_sc = cola_scores(orig_list)
    c_sc = cola_scores(t5_c)
    sims = tfidf_sim(orig_list, generated)
    h_ed = [get_edits(g, t) for g, t in zip(generated, t5_c)]
    r_ed = [get_edits(o, g) for o, g in zip(orig_list, gt_list)]
    return {
        'config':  name,
        'f0.5':    round(dist_f05(edit_type_dist(h_ed), edit_type_dist(r_ed)), 3),
        'cola_A':  round(np.mean(a_sc), 3),
        'cola_B':  round(np.mean(b_sc), 3),
        'cola_C':  round(np.mean(c_sc), 3),
        'sem_sim': round(np.mean(sims), 3),
        'B<A':     bool(np.mean(b_sc) < np.mean(a_sc)),
        'C>B':     bool(np.mean(c_sc) > np.mean(b_sc)),
    }


# ── Ablation run ───────────────────────────────────────────────────────────────

abl_rows = []

# Include main-run variant slices for direct comparison
MAIN_VARIANTS = ['zero_shot', 'one_shot', 'few_shot_3', 'few_shot_12', 'cot']
for v in MAIN_VARIANTS:
    cor_path = ROOT / f'data/corrected/pe_{v}_corrected.csv'
    if cor_path.exists():
        df_v = pd.read_csv(cor_path)
        abl_rows.append(eval_generated(v, df_v['generated'].tolist()[:N_ABL], abl_src, abl_gt))
        print(f'Evaluated (main slice): {v}  f0.5={abl_rows[-1]["f0.5"]}')

# New ablation configs
for name, (system, static_user, temperature) in NEW_CONFIGS.items():
    abl_path = ROOT / f'data/generated/pe_abl_{name}.csv'
    if abl_path.exists():
        print(f'Checkpoint: pe_abl_{name}.csv — loading.')
        df_abl = pd.read_csv(abl_path)
        generated = df_abl['generated'].tolist()
    else:
        print(f'\n--- {name} ---')
        try:
            generated = run_generation(system, static_user, abl_src,
                                       temperature=temperature, desc=name)
        except Exception as e:
            print(f'  FAILED: {e}')
            continue
        pd.DataFrame({
            'original':     abl_src,
            'ground_truth': abl_gt,
            'generated':    generated,
        }).to_csv(abl_path, index=False)
        print(f'  Saved → {abl_path}')
    row = eval_generated(name, generated, abl_src, abl_gt)
    abl_rows.append(row)
    print(f'  f0.5={row["f0.5"]}')

abl_df = pd.DataFrame(abl_rows).set_index('config').sort_values('f0.5', ascending=False)
out_path = ROOT / 'data/results/ablation_leaderboard.csv'
abl_df.to_csv(out_path)
print('\n=== ABLATION LEADERBOARD ===')
print(abl_df.to_string())
print(f'\nSaved → {out_path}')
