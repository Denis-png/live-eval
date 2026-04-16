"""
generation.py — CoT + expert-linguist generation on the full FCE dataset.

Run from Denis/:
    python summary/generation.py

Input:  data/fce/fce.m2
Output: data/summary/summary_generated.csv
        (columns: original, ground_truth, generated)

Combines CoT step-by-step format with the detailed (expert-linguist) instruction
wording from the ablation study.

Checkpoint: saves every SAVE_EVERY batches so the run can be resumed safely.
"""
import os, time, re, warnings
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from dotenv import load_dotenv
import anthropic

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent

load_dotenv(ROOT / '.env')
if not os.getenv('ANTHROPIC_API_KEY'):
    load_dotenv(ROOT.parent / 'Denis/.env')
assert os.getenv('ANTHROPIC_API_KEY'), 'ANTHROPIC_API_KEY not found'

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
        records.append({'original': ' '.join(src_tokens), 'ground_truth': ' '.join(tgt)})
    return pd.DataFrame(records)


M2_PATH = ROOT / 'data/fce/fce.m2'
df_fce  = load_m2(str(M2_PATH))
print(f'Loaded {len(df_fce)} FCE sentence pairs from {M2_PATH.name}')

if OUT_PATH.exists():
    print(f'Output already exists: {OUT_PATH.name} — delete to regenerate.')
    raise SystemExit(0)

# Resume from partial checkpoint (slice by row count, not set lookup)
done = 0
rows: list[dict] = []
if PARTIAL_PATH.exists():
    df_part = pd.read_csv(PARTIAL_PATH)
    rows = df_part.to_dict('records')
    done = len(rows)
    print(f'Resuming: {done} done, {len(df_fce) - done} remaining')

src_todo = df_fce['original'].tolist()[done:]
gt_todo  = df_fce['ground_truth'].tolist()[done:]

# ── LLM client ─────────────────────────────────────────────────────────────────

MODEL  = 'claude-haiku-4-5'
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# CoT with detailed (expert-linguist) instruction wording
SUMMARY_SYSTEM = (
    'You are an expert linguist specializing in grammatical error generation. '
    'For each numbered sentence below:\n'
    '  Step 1: Identify the grammatical error type '
    '(e.g. subject-verb agreement, article, preposition, verb tense, spelling).\n'
    '  Step 2: Create a completely new, original sentence on a different topic '
    'that exhibits the same category of grammatical error. '
    'The generated sentence MUST contain a grammatical error — '
    'do NOT produce a correct sentence.\n\n'
    'Use this exact format for each sentence:\n'
    '[N] Error type: <type>\n'
    '[N] Generated: <new sentence with error>\n\n'
    'Do NOT correct or rewrite the input sentence.'
)

# ── Batch utilities ────────────────────────────────────────────────────────────

BATCH_SIZE = 3
SLEEP_SEC  = 20
SAVE_EVERY = 50  # checkpoint every N batches


def make_batch_input(sentences: list[str]) -> str:
    return '\n'.join(f'{i+1}. {s}' for i, s in enumerate(sentences))


def parse_cot_output(text: str, expected: int) -> list[str]:
    generated = []
    for line in text.strip().splitlines():
        m = re.match(r'^\[[0-9]+\]\s*Generated:\s*(.+)', line.strip())
        if m:
            generated.append(m.group(1).strip())
    if len(generated) == expected:
        return generated
    generated = [
        line.split('Generated:', 1)[1].strip()
        for line in text.strip().splitlines() if 'Generated:' in line
    ]
    if len(generated) == expected:
        return generated
    raise ValueError(f'CoT parse failed. Expected {expected}, got {len(generated)}:\n{text}')


# ── Generation loop ────────────────────────────────────────────────────────────

batches = [
    (src_todo[i:i+BATCH_SIZE], gt_todo[i:i+BATCH_SIZE])
    for i in range(0, len(src_todo), BATCH_SIZE)
]

for batch_idx, (batch_src, batch_gt) in enumerate(tqdm(batches, desc='FCE generation')):
    batch_input = make_batch_input(batch_src)
    success = False

    for attempt in range(7):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=[{
                    'type': 'text',
                    'text': SUMMARY_SYSTEM,
                    'cache_control': {'type': 'ephemeral'},
                }],
                messages=[{'role': 'user', 'content': batch_input}],
            )
            generated = parse_cot_output(response.content[0].text, len(batch_src))
            for o, g, gen in zip(batch_src, batch_gt, generated):
                rows.append({'original': o, 'ground_truth': g, 'generated': gen})
            success = True
            break
        except ValueError:
            time.sleep(10 * (2 ** attempt))
        except anthropic.RateLimitError:
            time.sleep(60 * (2 ** attempt))

    if not success:
        for o, g in zip(batch_src, batch_gt):
            rows.append({'original': o, 'ground_truth': g, 'generated': '[GENERATION FAILED]'})
        print(f'Warning: batch {done // BATCH_SIZE + batch_idx} failed')

    time.sleep(SLEEP_SEC)

    if (batch_idx + 1) % SAVE_EVERY == 0:
        pd.DataFrame(rows).to_csv(PARTIAL_PATH, index=False)
        print(f'  Checkpoint: {len(rows)} rows saved')

# ── Final save ─────────────────────────────────────────────────────────────────

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_PATH, index=False)
if PARTIAL_PATH.exists():
    PARTIAL_PATH.unlink()

failed = (df_out['generated'] == '[GENERATION FAILED]').sum()
print(f'\nSaved {len(df_out)} rows → {OUT_PATH}  ({failed} failed)')
