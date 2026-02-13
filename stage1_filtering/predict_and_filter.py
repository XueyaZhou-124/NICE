"""
Predict maternal contamination score per read and optionally filter BAM to embryo reads (score <= threshold).
Outputs: (1) CSV with header and C-score per read; (2) optional BAM with reads where C-score <= threshold.
Run from project root: PYTHONPATH=. python -m stage1_decent.predict_and_filter --bam_path <sample.id.bam> ...
Or from stage1_decent: python predict_and_filter.py --bam_path <sample.id.bam> ...
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import pysam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from .utils import extract_reads, split_methy, conv_onehot
    from .model import DISMIR_deep
except ImportError:
    from utils import extract_reads, split_methy, conv_onehot
    from model import DISMIR_deep


def read_preprocess(reads_path):
    """Parse .reads file; return seq, methy, headers."""
    seq, methy, headers = [], [], []
    with open(reads_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 4:
                headers.append(parts[0])
                seq.append(parts[3])
                methy.append(parts[4])
    return seq, methy, headers


def predict_score(bam_path, reads_dir, model_path, res_dir, res_file=None, remove_reads_after=True):
    """
    For each read in BAM: get score (probability of maternal contamination).
    C-score in output = 1 - model_output (embryo probability when binary label 1 = embryo).
    Returns path to CSV: header, C-score.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    basename = os.path.basename(bam_path).replace('.bam', '')
    os.makedirs(res_dir, exist_ok=True)
    out_csv = res_file or os.path.join(res_dir, basename + '.csv')
    if res_file and os.path.exists(res_file):
        return res_file
    if not res_file and os.path.exists(out_csv):
        return out_csv

    reads_path = os.path.join(reads_dir, basename + '.reads')
    if not os.path.exists(reads_path):
        extract_reads(bam_path, reads_dir)

    seq, methy, headers = read_preprocess(reads_path)
    seq_lstm = split_methy(seq, methy)
    seq_one_hot = conv_onehot(seq_lstm)
    loader = DataLoader(TensorDataset(seq_one_hot), batch_size=10000)

    model = DISMIR_deep(n_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    scores = []
    with torch.no_grad():
        for (inputs,) in tqdm(loader):
            inputs = inputs.to(device).float()
            out = model(inputs.permute(0, 2, 1))
            scores.append(out[:, 0].cpu().numpy())
    scores = np.concatenate(scores)
    # C-score: maternal contamination probability (original code used 1 - output as embryo score)
    c_score = 1 - scores
    pd.DataFrame({'header': headers, 'C-score': c_score}).to_csv(out_csv, index=False)
    if remove_reads_after and os.path.exists(reads_path):
        os.remove(reads_path)
    return out_csv


def filter_bam_by_score(bam_path, csv_path, threshold, out_bam_path=None):
    """Write BAM containing only reads with C-score <= threshold (embryo reads)."""
    if out_bam_path is None:
        base = bam_path.replace('.bam', '')
        out_bam_path = base + '_' + str(threshold).replace('.', '') + '.bam'
    df = pd.read_csv(csv_path, dtype={'header': str, 'C-score': np.float64})
    keep = set(df.loc[df['C-score'] <= threshold, 'header'].astype(str))
    with pysam.AlignmentFile(bam_path, 'rb') as fin, \
         pysam.AlignmentFile(out_bam_path, 'wb', header=fin.header) as fout:
        for read in fin:
            if read.query_name in keep:
                fout.write(read)
    return out_bam_path


def main():
    parser = argparse.ArgumentParser(description='Predict read-level score and optionally filter BAM.')
    parser.add_argument('--bam_path', required=True, help='Path to .id.bam')
    parser.add_argument('--model_path', required=True, help='Path to trained .pth')
    parser.add_argument('--reads_dir', default='.', help='Temporary directory for .reads')
    parser.add_argument('--res_dir', default='.', help='Directory for output CSV')
    parser.add_argument('--res_file', default=None, help='Exact path for output CSV (overrides res_dir)')
    parser.add_argument('--threshold', type=float, default=0.2,
                        help='Keep reads with C-score <= this (embryo). Use 1 to skip filtering.')
    parser.add_argument('--no_filter', action='store_true', help='Only output CSV, do not write filtered BAM')
    args = parser.parse_args()

    csv_path = predict_score(
        args.bam_path, args.reads_dir, args.model_path,
        args.res_dir, args.res_file, remove_reads_after=True
    )
    if not args.no_filter and args.threshold < 1:
        out_bam = filter_bam_by_score(args.bam_path, csv_path, args.threshold)
        print('Filtered BAM written:', out_bam)
    print('Scores CSV:', csv_path)


if __name__ == '__main__':
    main()
