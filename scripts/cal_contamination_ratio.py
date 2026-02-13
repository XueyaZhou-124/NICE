"""
Optional: estimate per-sample maternal contamination ratio from DECENT score CSV.
Finds the mixture proportion that maximizes log-likelihood over read scores.
Usage: python cal_contamination_ratio.py --res_dir <dir_with_*.csv> [--out_file ratio.txt]
"""

import os
import argparse
import numpy as np
import pandas as pd


def cal_ratio(csv_path, gaps=1000):
    """Return best contamination (maternal) proportion in [0,1]."""
    df = pd.read_csv(csv_path)
    score_col = 'C-score' if 'C-score' in df.columns else [c for c in df.columns if 'score' in c.lower()][0]
    likelihood_1 = df[score_col].astype(np.float32).values
    likelihood_2 = 1 - likelihood_1
    best_sum, best_gap = -np.inf, 0.0
    for gap in np.linspace(0, 1, gaps):
        s = np.log10(gap * likelihood_1 + (1 - gap) * likelihood_2)
        total = np.nansum(s)
        if total > best_sum:
            best_sum, best_gap = total, round(gap, 3)
    return best_gap, best_sum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', required=True, help='Directory containing *\.csv score files')
    parser.add_argument('--out_file', default=None, help='Append results here (default: res_dir/ratio.txt)')
    args = parser.parse_args()
    out = args.out_file or os.path.join(args.res_dir, 'ratio.txt')
    for f in sorted(os.listdir(args.res_dir)):
        if not f.endswith('.csv'):
            continue
        path = os.path.join(args.res_dir, f)
        sample = os.path.splitext(f)[0]
        gap, loglik = cal_ratio(path)
        with open(out, 'a') as fp:
            fp.write(f'{sample}\t{gap}\t{loglik}\n')
        print(f'Sample: {sample}, Best gap: {gap}, log10_sum: {loglik}')
    print('Written to', out)


if __name__ == '__main__':
    main()
