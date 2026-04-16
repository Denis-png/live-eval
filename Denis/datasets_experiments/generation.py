"""
generation.py — CoT generation for each dataset tier.

Run from Denis/datasets_experiments/:
    python generation.py

Input:  data/sampled/ds_{name}_sample.csv
Output: data/generated/ds_{name}_generated.csv  (columns: original, ground_truth, tier, generated)

Skips datasets whose output CSV already exists (checkpoint).
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

(ROOT / 'data/generated').mkdir(parents=True, exist_ok=True)

DATASETS = ['conll14', 'fce', 'c4_200m']

# ── LLM client ─────────────────────────────────────────────────────────────────

MODEL  = 'claude-haiku-4-5'
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# ── CoT prompt ─────────────────────────────────────────────────────────────────

COT_SYSTEM = (
    'You are a grammatical error generator. For each numbered sentence below:\n'
    '  Step 1: Identify the grammatical error type (e.g. subject-verb agreement, article, spelling).\n'
    '  Step 2: Write a completely NEW sentence on a different topic with the same error type.\n'
    '  The generated sentence MUST contain a grammatical error — do NOT produce a correct sentence.\n\n'
    'Use this exact format for each sentence:\n'
    '[N] Error type: <type>\n'
    '[N] Generated: <new sentence>\n\n'
    'Do NOT correct or rewrite the input sentence.'
)

# ── Batch utilities ────────────────────────────────────────────────────────────

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


def run_cot(sentences: list[str], batch_size: int = 3,
            sleep: int = 20, desc: str = 'CoT') -> list[str]:
    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
    results, failed = [], []
    for idx, batch in enumerate(tqdm(batches, desc=desc)):
        batch_input = make_batch_input(batch)
        success = False
        for attempt in range(7):
            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=[{
                        'type': 'text',
                        'text': COT_SYSTEM,
                        'cache_control': {'type': 'ephemeral'},
                    }],
                    messages=[{'role': 'user', 'content': batch_input}],
                )
                out = response.content[0].text
                results.extend(parse_cot_output(out, len(batch)))
                success = True
                break
            except ValueError:
                time.sleep(10 * (2 ** attempt))
            except anthropic.RateLimitError:
                time.sleep(60 * (2 ** attempt))
        if not success:
            results.extend(['[GENERATION FAILED]'] * len(batch))
            failed.append(idx)
        time.sleep(sleep)
    if failed:
        print(f'Warning: {len(failed)} batch(es) failed — idx {failed}')
    return results


# ── Run generation per dataset ─────────────────────────────────────────────────

def main() -> None:
    for name in DATASETS:
        out_path = ROOT / f'data/generated/ds_{name}_generated.csv'
        if out_path.exists():
            print(f'Checkpoint: {out_path.name} — skipping.')
            continue
        sample_path = ROOT / f'data/sampled/ds_{name}_sample.csv'
        if not sample_path.exists():
            print(f'Missing: {sample_path.name} — run sampling.py first.')
            continue

        df = pd.read_csv(sample_path)
        print(f'\n--- CoT generation: {name} ({df["tier"].iloc[0]}) ---')
        generated = run_cot(df['original'].tolist(), desc=name)

        out_df = pd.DataFrame({
            'original':     df['original'],
            'ground_truth': df['ground_truth'],
            'tier':         df['tier'],
            'generated':    generated,
        })
        out_df.to_csv(out_path, index=False)
        print(f'Generated {len(generated)} sentences → {out_path}')


if __name__ == '__main__':
    main()
