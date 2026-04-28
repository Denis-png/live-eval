"""
sampling.py — Sample from C4_200M TSV files and save checkpoint.

Run from Denis/prompt_engineering/:
    python sampling.py

Output: data/sampled/pe_sample.csv  (columns: original, ground_truth)
"""
import os, glob, warnings
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent   # Denis/

DEMO_MODE  = True
N_PER_FILE = 10 if DEMO_MODE else 100
C4_PATTERN = str(ROOT / 'data/C4_200M/C4_200M.tsv-*')
OUT_PATH   = ROOT / 'data/sampled/pe_sample.csv'
SEED       = 42
CHUNKSIZE  = 5_000


def sample_from_tsv(pattern: str, n_rows: int, seed: int = SEED,
                    chunksize: int = CHUNKSIZE) -> pd.DataFrame:
    """Memory-safe random sample from TSV glob (reservoir per chunk)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'No files matched: {pattern}')

    chunks = []
    for path in tqdm(files, desc='Sampling files'):
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


def main() -> None:
    if OUT_PATH.exists():
        df = pd.read_csv(OUT_PATH)
        print(f'Checkpoint found: {OUT_PATH} ({len(df)} rows) — skipping re-sample.')
    else:
        print(f'Mode: {"demo" if DEMO_MODE else "full"} — {N_PER_FILE} rows/file × 10 files')
        df = sample_from_tsv(C4_PATTERN, n_rows=N_PER_FILE)
        df = df[['original', 'ground_truth']]
        df.to_csv(OUT_PATH, index=False)
        print(f'Saved {len(df)} rows → {OUT_PATH}')

    print(df.head(3).to_string())


if __name__ == '__main__':
    main()
