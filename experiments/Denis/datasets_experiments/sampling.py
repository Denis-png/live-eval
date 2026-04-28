"""
sampling.py — Load and sample from CoNLL-2014, FCE, and C4_200M datasets.

Run from Denis/datasets_experiments/:
    python sampling.py

Output (one checkpoint per dataset):
    data/sampled/ds_conll14_sample.csv
    data/sampled/ds_fce_sample.csv
    data/sampled/ds_c4_200m_sample.csv

All files: columns original, ground_truth
"""
import glob, warnings
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

ROOT     = Path(__file__).parent.parent
SEED     = 42
N_SAMPLE = 50    # per dataset
CHUNKSIZE = 5_000

(ROOT / 'data/sampled').mkdir(parents=True, exist_ok=True)

# ── M2 parser ─────────────────────────────────────────────────────────────────

def load_m2(path: str, annotator_id: int = 0) -> pd.DataFrame:
    """Parse .m2 file → DataFrame(original, ground_truth).

    Skips sentences where the selected annotator made no edits (already correct).
    """
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
            start, end   = int(span[0]), int(span[1])
            correction   = parts[2]
            ann          = int(parts[5])
            if ann == annotator_id and correction != '-NONE-':
                edits.append((start, end, correction))
        if not edits:
            continue
        tgt = src_tokens[:]
        for start, end, correction in sorted(edits, key=lambda x: -x[0]):
            tgt[start:end] = correction.split() if correction else []
        records.append({
            'original':     ' '.join(src_tokens),
            'ground_truth': ' '.join(tgt),
        })

    df = pd.DataFrame(records)
    print(f'  Loaded {len(df)} erroneous pairs from {path}')
    return df


# ── TSV sampler ────────────────────────────────────────────────────────────────

def sample_from_tsv(pattern: str, n_rows: int, seed: int = SEED,
                    chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files matched: {pattern}')
    chunks = []
    for path in tqdm(files, desc='Sampling TSV'):
        reservoir = []
        for chunk in pd.read_csv(
            path, sep='\t', header=None,
            names=['original', 'ground_truth'],
            chunksize=chunksize,
        ):
            reservoir.append(chunk.sample(n=min(n_rows, len(chunk)), random_state=seed))
        file_df = pd.concat(reservoir).sample(n=n_rows, random_state=seed).reset_index(drop=True)
        chunks.append(file_df)
        del reservoir, file_df
    return pd.concat(chunks, ignore_index=True)


# ── Dataset configs ────────────────────────────────────────────────────────────

DATASETS = {
    'conll14': {'type': 'm2',  'path': str(ROOT / 'data/conll14/conll14st-test.m2')},
    'fce':     {'type': 'm2',  'path': str(ROOT / 'data/fce/fce.m2')},
    'c4_200m': {'type': 'tsv', 'path': str(ROOT / 'data/C4_200M/C4_200M.tsv-*')},
}

TIER_LABELS = {
    'conll14': 'Test set',
    'fce':     'Benchmark',
    'c4_200m': 'General',
}


def main() -> None:
    for name, cfg in DATASETS.items():
        out_path = ROOT / f'data/sampled/ds_{name}_sample.csv'
        if out_path.exists():
            df = pd.read_csv(out_path)
            print(f'Checkpoint: {out_path.name} ({len(df)} rows) — skipping.')
            continue

        print(f'\nLoading {name}...')
        if cfg['type'] == 'm2':
            df = load_m2(cfg['path'])
            df = df.sample(n=min(N_SAMPLE, len(df)), random_state=SEED).reset_index(drop=True)
        else:
            n_files    = len(glob.glob(cfg['path']))
            n_per_file = max(1, N_SAMPLE // n_files)
            df = sample_from_tsv(cfg['path'], n_rows=n_per_file)
            df = df.sample(n=min(N_SAMPLE, len(df)), random_state=SEED).reset_index(drop=True)

        df = df[['original', 'ground_truth']].copy()
        df['tier'] = TIER_LABELS[name]
        df.to_csv(out_path, index=False)
        print(f'  → {len(df)} rows saved to {out_path}')

    print('\nSample sizes:')
    for name in DATASETS:
        p = ROOT / f'data/sampled/ds_{name}_sample.csv'
        if p.exists():
            print(f'  {name}: {len(pd.read_csv(p))} rows')


if __name__ == '__main__':
    main()
