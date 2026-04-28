"""
generation.py — Run all 5 prompt variants on the C4_200M sample.

Run from Denis/prompt_engineering/:
    python generation.py

Input:  data/sampled/pe_sample.csv
Output: data/generated/pe_{variant}.csv  (columns: original, ground_truth, generated)

Variants: zero_shot, one_shot, few_shot_3, few_shot_12, cot
Skips variants whose output CSV already exists (checkpoint).
"""
import os, json, time, re, warnings
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
assert os.getenv('ANTHROPIC_API_KEY'), 'ANTHROPIC_API_KEY not found — check Denis/.env'

(ROOT / 'data/generated').mkdir(parents=True, exist_ok=True)

# ── Data ───────────────────────────────────────────────────────────────────────

df  = pd.read_csv(ROOT / 'data/sampled/pe_sample.csv')
src = df['original'].tolist()
gt  = df['ground_truth'].tolist()
print(f'Loaded {len(df)} rows from data/sampled/pe_sample.csv')

# ── LLM client ─────────────────────────────────────────────────────────────────
# cache_control markers are added to static content blocks below.
# Note: Haiku 4.5 requires ≥4096 tokens in the cached prefix to actually save
# a cache entry. Current prompts are 100–500 tokens so cache writes are charged
# at 1.25× and reads will silently miss until that threshold is reached.
# Use claude-sonnet-4-5 (1024-token minimum) if you want caching to fire sooner.

MODEL  = 'claude-haiku-4-5'
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))


def call_llm(system: str, static_user: str, dynamic_user: str,
             temperature: float = 0.7) -> str:
    """Single API call. static_user (examples/prefix) is marked for caching;
    dynamic_user (the numbered batch) is not."""
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


# ── Shared examples ────────────────────────────────────────────────────────────

with open(ROOT / 'data/gec_error_types.json') as f:
    error_types = json.load(f)['error_types']

with open(ROOT / 'data/gec_few_shot_examples.json') as f:
    few_shot_data = json.load(f)

demonstrations = few_shot_data['demonstrations']

# ── Batch utilities ────────────────────────────────────────────────────────────

BATCH_SIZE = 5
SLEEP_SEC  = 10


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
    raise ValueError(
        f'Expected {expected} lines, got {len(lines)} numbered / {len(fallback)} plain:\n{text}'
    )


def invoke_with_retry(system: str, static_user: str, sentences: list[str],
                      temperature: float = 0.7,
                      max_retries: int = 7, base_delay: int = 10) -> list[str]:
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
            wait = 60 * (2 ** attempt)
            print(f'\nRate limit — retrying in {wait}s...')
            time.sleep(wait)
    raise RuntimeError(f'Failed after {max_retries} retries. Last: {last_error}')


def run_generation(system: str, static_user: str, sentences: list[str],
                   temperature: float = 0.7,
                   batch_size: int = BATCH_SIZE, sleep: int = SLEEP_SEC,
                   desc: str = 'Generating') -> list[str]:
    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
    results = []
    for batch in tqdm(batches, desc=desc):
        results.extend(invoke_with_retry(system, static_user, batch, temperature))
        time.sleep(sleep)
    return results


# ── CoT utilities ──────────────────────────────────────────────────────────────

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


def run_cot(system: str, static_user: str, sentences: list[str],
            batch_size: int = 3, sleep: int = 20, desc: str = 'CoT') -> list[str]:
    batches = [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]
    results, failed = [], []
    for idx, batch in enumerate(tqdm(batches, desc=desc)):
        dynamic_user = make_batch_input(batch)
        success = False
        for attempt in range(7):
            try:
                out = call_llm(system, static_user, dynamic_user)
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


# ── Prompt variants ────────────────────────────────────────────────────────────

ZS_SYSTEM = (
    'You are a grammatical error generator. '
    'For each sentence below, write a completely NEW sentence on a different topic '
    'that contains the same type of grammatical error. '
    'The generated sentence MUST contain a grammatical error — do NOT produce a correct sentence. '
    'Do NOT correct the given sentence — create an entirely new one. '
    'Return only the generated sentences, one per line, preserving the original numbering.'
)

ONE_SHOT_STATIC = (
    'Example:\n'
    'Input:  1. The group of students are going to the library.\n'
    'Output: 1. The team of engineers was arguing about their blueprints.\n\n'
    'Now generate for:\n'
)

FS_SYSTEM = (
    'You are a grammatical error generator. '
    'For each sentence below, write a completely NEW sentence on a different topic '
    'that contains the same type of grammatical error as the input. '
    'The generated sentence MUST contain a grammatical error — do NOT produce a correct sentence. '
    'Do NOT correct the given sentence.\n'
    'Return only the generated sentences, one per line, preserving the numbering.'
)

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


def make_few_shot_block(pairs: list[tuple[str, str]]) -> str:
    lines = []
    for i, (inp, out) in enumerate(pairs, 1):
        lines += [f'Example {i}:', f'Input:  {inp}', f'Output: {out}', '']
    return '\n'.join(lines) + 'Now generate for:\n'


pairs_3 = [
    (demonstrations[k]['input'], demonstrations[k]['output'])
    for k in ('subject_verb_agreement', 'article', 'verb_tense')
]
pairs_12 = [
    (demonstrations[et['name']]['input'], demonstrations[et['name']]['output'])
    for et in error_types
]

FS3_STATIC  = make_few_shot_block(pairs_3)
FS12_STATIC = make_few_shot_block(pairs_12)

# ── Run variants ───────────────────────────────────────────────────────────────
# Each entry: (system, static_user, runner_fn, extra_kwargs)
VARIANTS: dict[str, tuple] = {
    'zero_shot':   (ZS_SYSTEM,  '',            run_generation, {}),
    'one_shot':    (ZS_SYSTEM,  ONE_SHOT_STATIC, run_generation, {}),
    'few_shot_3':  (FS_SYSTEM,  FS3_STATIC,    run_generation, {'batch_size': 3, 'sleep': 15}),
    'few_shot_12': (FS_SYSTEM,  FS12_STATIC,   run_generation, {'batch_size': 3, 'sleep': 20}),
    'cot':         (COT_SYSTEM, '',            run_cot,        {'batch_size': 3, 'sleep': 20}),
}


def main() -> None:
    for name, (system, static_user, fn, kwargs) in VARIANTS.items():
        out_path = ROOT / f'data/generated/pe_{name}.csv'
        if out_path.exists():
            print(f'Checkpoint: {out_path.name} — skipping.')
            continue
        print(f'\n--- {name} ---')
        generated = fn(system, static_user, src, desc=name, **kwargs)
        pd.DataFrame({
            'original':     src,
            'ground_truth': gt,
            'generated':    generated,
        }).to_csv(out_path, index=False)
        print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
