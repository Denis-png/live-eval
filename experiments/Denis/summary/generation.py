"""
generation.py — CoT + expert-linguist generation on the full FCE dataset.

Run from Denis/:
    python summary/generation.py [--model haiku|nemotron]

    --model haiku     claude-haiku-4-5 via Anthropic API  (default)
    --model nemotron  nvidia/llama-3.1-nemotron-ultra-253b-v1:free via OpenRouter

Input:  data/fce/fce.m2
Output: data/summary/summary_generated.csv
        (columns: original, fce_ground_truth, generated, gen_ground_truth)

Pipeline per batch:
  1. Generate: identify error type → write B1-learner sentence → write its correction
  2. Judge:    filter trivial errors and invalid corrections (LLM-as-judge)

Checkpoint: saves every SAVE_EVERY batches so the run can be resumed safely.
Requires:  anthropic, openai  (openai only needed for --model nemotron)
"""
import argparse, os, time, re, warnings
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='FCE benchmark generation')
parser.add_argument(
    '--model', choices=['haiku', 'nemotron'], default='haiku',
    help='Model backend: haiku (Anthropic) or nemotron (OpenRouter free)',
)
args = parser.parse_args()

# ── Per-model config ───────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'haiku': {
        'model_id':         'claude-haiku-4-5',
        'batch_size':       3,
        'sleep_sec':        20,
        'gen_max_tokens':   1024,
        'judge_max_tokens': 256,
    },
    'nemotron': {
        # Free tier on OpenRouter — conservative limits to avoid 429s
        'model_id':         'nvidia/llama-3.1-nemotron-ultra-253b-v1:free',
        'batch_size':       1,
        'sleep_sec':        10,
        'gen_max_tokens':   1024,
        'judge_max_tokens': 256,
    },
}

cfg        = MODEL_CONFIGS[args.model]
BATCH_SIZE = cfg['batch_size']
SLEEP_SEC  = cfg['sleep_sec']
SAVE_EVERY = 50

print(f'Using model: {args.model} ({cfg["model_id"]})')

# ── Environment + clients ──────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent

load_dotenv(ROOT / '.env')
if not os.getenv('ANTHROPIC_API_KEY'):
    load_dotenv(ROOT.parent / 'Denis/.env')

if args.model == 'haiku':
    import anthropic as _anthropic
    assert os.getenv('ANTHROPIC_API_KEY'), 'ANTHROPIC_API_KEY not found'
    _anthropic_client = _anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    _openrouter_client = None
else:
    try:
        from openai import OpenAI as _OpenAI, RateLimitError as _OAIRateLimitError
    except ImportError:
        raise SystemExit('openai package required for --model nemotron: pip install openai')
    assert os.getenv('OPENROUTER_API_KEY'), 'OPENROUTER_API_KEY not found'
    _openrouter_client = _OpenAI(
        base_url='https://openrouter.ai/api/v1',
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )
    _anthropic_client = None

(ROOT / 'data/summary').mkdir(parents=True, exist_ok=True)

OUT_PATH     = ROOT / 'data/summary/summary_generated.csv'
PARTIAL_PATH = ROOT / 'data/summary/summary_generated_partial.csv'

# ── Load full FCE M2 ───────────────────────────────────────────────────────────

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
        records.append({'original': ' '.join(src_tokens), 'fce_ground_truth': ' '.join(tgt)})
    return pd.DataFrame(records)


M2_PATH = ROOT / 'data/fce/fce.m2'
df_fce  = load_m2(str(M2_PATH))
print(f'Loaded {len(df_fce)} FCE sentence pairs from {M2_PATH.name}')

if OUT_PATH.exists():
    print(f'Output already exists: {OUT_PATH.name} — delete to regenerate.')
    raise SystemExit(0)

# Resume from partial checkpoint
done = 0
rows: list[dict] = []
judge_filtered_total = 0
if PARTIAL_PATH.exists():
    df_part = pd.read_csv(PARTIAL_PATH)
    rows = df_part.to_dict('records')
    done = len(rows)
    print(f'Resuming: {done} done, {len(df_fce) - done} remaining')

src_todo = df_fce['original'].tolist()[done:]
gt_todo  = df_fce['fce_ground_truth'].tolist()[done:]

# ── Prompts ────────────────────────────────────────────────────────────────────

SUMMARY_SYSTEM = (
    'You are an expert linguist specializing in grammatical error generation. '
    'For each numbered sentence below:\n'
    '  Step 1: Identify the grammatical error type '
    '(e.g. subject-verb agreement, article, preposition, verb tense, spelling).\n'
    '  Step 2: Write a new, original sentence a B1-level foreign learner would produce '
    'on a different topic, exhibiting the same error type plus typical L1-transfer errors. '
    'The generated sentence MUST contain a grammatical error — '
    'do NOT produce a correct sentence.\n'
    '  Step 3: Write the fully corrected version of the sentence from Step 2 '
    '(fix all grammatical errors so it reads as natural, fluent English).\n\n'
    'Use this exact format for each sentence:\n'
    '[N] Error type: <type>\n'
    '[N] Generated: <new sentence with error>\n'
    '[N] Ground truth: <corrected version of Generated>\n\n'
    'Do NOT correct or rewrite the input sentence.'
)

JUDGE_SYSTEM = (
    'You are a GEC (grammatical error correction) quality evaluator. '
    'For each numbered item you receive an erroneous sentence and its proposed correction.\n\n'
    'Assess two criteria:\n'
    '  Redundancy: "trivial" if the error is too obvious or uninteresting for GEC evaluation '
    '(e.g. a single missing full-stop, one clearly wrong article in isolation, '
    'a trivial misspelling of one very common word with no other errors). '
    '"valid" otherwise.\n'
    '  Correction: "correct" if the proposed correction is grammatically correct and '
    'natural English with no remaining errors. "incorrect" otherwise.\n\n'
    'Use this exact format:\n'
    '[N] Redundancy: trivial/valid\n'
    '[N] Correction: correct/incorrect'
)

# ── Unified API call ───────────────────────────────────────────────────────────

def call_model(system: str, user: str, max_tokens: int, temperature: float) -> str:
    """Call the selected model; returns the text response."""
    if args.model == 'haiku':
        response = _anthropic_client.messages.create(
            model=cfg['model_id'],
            max_tokens=max_tokens,
            temperature=temperature,
            system=[{
                'type': 'text',
                'text': system,
                'cache_control': {'type': 'ephemeral'},
            }],
            messages=[{'role': 'user', 'content': user}],
        )
        return response.content[0].text
    else:
        response = _openrouter_client.chat.completions.create(
            model=cfg['model_id'],
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {'role': 'system', 'content': system},
                {'role': 'user',   'content': user},
            ],
        )
        return response.choices[0].message.content


def _is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__
    return 'RateLimitError' in name or getattr(exc, 'status_code', None) == 429


# ── Batch utilities ────────────────────────────────────────────────────────────

def make_batch_input(sentences: list[str]) -> str:
    return '\n'.join(f'{i+1}. {s}' for i, s in enumerate(sentences))


def parse_gen_output(text: str, expected: int) -> list[tuple[str, str]]:
    """Parse Step 2+3 output → list of (generated, gen_ground_truth) tuples."""
    generated: dict[int, str] = {}
    ground_truths: dict[int, str] = {}
    for line in text.strip().splitlines():
        m = re.match(r'^\[([0-9]+)\]\s*Generated:\s*(.+)', line.strip())
        if m:
            generated[int(m.group(1))] = m.group(2).strip()
        m = re.match(r'^\[([0-9]+)\]\s*Ground\s*truth:\s*(.+)', line.strip())
        if m:
            ground_truths[int(m.group(1))] = m.group(2).strip()

    results = [
        (generated[i], ground_truths[i])
        for i in range(1, expected + 1)
        if i in generated and i in ground_truths
    ]
    if len(results) == expected:
        return results
    raise ValueError(
        f'Gen parse failed. Expected {expected}, got {len(results)}:\n{text}'
    )


def parse_judge_output(text: str, expected: int) -> list[bool]:
    """Parse judge response → list of pass booleans (True = keep)."""
    redundancy: dict[int, bool] = {}
    correction: dict[int, bool] = {}
    for line in text.strip().splitlines():
        m = re.match(r'^\[([0-9]+)\]\s*Redundancy:\s*(trivial|valid)', line.strip(), re.IGNORECASE)
        if m:
            redundancy[int(m.group(1))] = m.group(2).lower() == 'valid'
        m = re.match(r'^\[([0-9]+)\]\s*Correction:\s*(correct|incorrect)', line.strip(), re.IGNORECASE)
        if m:
            correction[int(m.group(1))] = m.group(2).lower() == 'correct'

    results = [
        redundancy.get(i, True) and correction.get(i, True)
        for i in range(1, expected + 1)
    ]
    if len(results) == expected:
        return results
    return [True] * expected


def judge_batch(generated_list: list[str], gt_list: list[str]) -> list[bool]:
    """Run LLM-as-judge; returns pass/fail per sentence."""
    items = '\n'.join(
        f'{i+1}. Error: {g}\n   Correction: {gt}'
        for i, (g, gt) in enumerate(zip(generated_list, gt_list))
    )
    for attempt in range(4):
        try:
            text = call_model(
                JUDGE_SYSTEM, items,
                max_tokens=cfg['judge_max_tokens'],
                temperature=0,
            )
            return parse_judge_output(text, len(generated_list))
        except Exception as exc:
            time.sleep(60 * (2 ** attempt) if _is_rate_limit(exc) else 10 * (2 ** attempt))
    return [True] * len(generated_list)


# ── Generation loop ────────────────────────────────────────────────────────────

batches = [
    (src_todo[i:i+BATCH_SIZE], gt_todo[i:i+BATCH_SIZE])
    for i in range(0, len(src_todo), BATCH_SIZE)
]

for batch_idx, (batch_src, batch_fce_gt) in enumerate(tqdm(batches, desc='FCE generation')):
    batch_input = make_batch_input(batch_src)
    success = False

    for attempt in range(7):
        try:
            text  = call_model(
                SUMMARY_SYSTEM, batch_input,
                max_tokens=cfg['gen_max_tokens'],
                temperature=1.0,
            )
            pairs = parse_gen_output(text, len(batch_src))

            # ── LLM-as-judge ──────────────────────────────────────────────────
            gen_sentences = [p[0] for p in pairs]
            gen_gts       = [p[1] for p in pairs]
            verdicts      = judge_batch(gen_sentences, gen_gts)

            n_filtered = verdicts.count(False)
            if n_filtered:
                judge_filtered_total += n_filtered
                tqdm.write(f'  Judge filtered {n_filtered}/{len(verdicts)} in batch {batch_idx}')

            for o, fce_gt, (gen, gen_gt), keep in zip(
                batch_src, batch_fce_gt, pairs, verdicts
            ):
                if keep:
                    rows.append({
                        'original':         o,
                        'fce_ground_truth': fce_gt,
                        'generated':        gen,
                        'gen_ground_truth': gen_gt,
                    })

            success = True
            break
        except ValueError:
            time.sleep(10 * (2 ** attempt))
        except Exception as exc:
            time.sleep(60 * (2 ** attempt) if _is_rate_limit(exc) else 10 * (2 ** attempt))

    if not success:
        for o, fce_gt in zip(batch_src, batch_fce_gt):
            rows.append({
                'original':         o,
                'fce_ground_truth': fce_gt,
                'generated':        '[GENERATION FAILED]',
                'gen_ground_truth': '[GENERATION FAILED]',
            })
        tqdm.write(f'Warning: batch {done // BATCH_SIZE + batch_idx} failed after all retries')

    time.sleep(SLEEP_SEC)

    if (batch_idx + 1) % SAVE_EVERY == 0:
        pd.DataFrame(rows).to_csv(PARTIAL_PATH, index=False)
        tqdm.write(f'  Checkpoint: {len(rows)} rows kept, {judge_filtered_total} filtered so far')

# ── Final save ─────────────────────────────────────────────────────────────────

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_PATH, index=False)
if PARTIAL_PATH.exists():
    PARTIAL_PATH.unlink()

failed   = (df_out['generated'] == '[GENERATION FAILED]').sum()
total_in = len(df_fce) - done
print(f'\nSaved {len(df_out)} rows → {OUT_PATH}')
print(f'  Generation failures : {failed}')
print(f'  Judge filtered      : {judge_filtered_total} / {total_in} ({100*judge_filtered_total/max(total_in,1):.1f}%)')
