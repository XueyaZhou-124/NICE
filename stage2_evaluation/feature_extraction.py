"""
Stage2: Extract six feature types from purified BAM / singleC / external CNA.
Each feature type is written as a CSV (one row per sample).
Requires label.csv in save_path (created from --sample_info when running target_methy first).
Run: python feature_extraction.py --target_methy_path ... --gwm_path ... --sample_info ... --bam_path ... --save_path ...
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import pysam
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm

warnings.filterwarnings('ignore')

PHENO_LABEL_COLS = ['Morphology', 'typeMorphology', 'typeMorphology2']


def _read_label(save_path):
    """Load label DataFrame from save_path/label.csv."""
    path = os.path.join(save_path, 'label.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'label.csv not found in {save_path}. Run target_methy first or provide sample_info.')
    return pd.read_csv(path, index_col=0)


def _normalize_sample_id(sample):
    """Normalize sample IDs across files, e.g. PBAT_E12_B2_id -> PBAT_E12_B2."""
    sample = str(sample).strip()
    if sample.endswith('_id'):
        sample = sample[:-3]
    return sample


def _resolve_singlec_path(singlec_dir, sample, singlec_suffix):
    """Resolve singleC path from either nested or flat layout."""
    nested_path = os.path.join(singlec_dir, sample, sample + singlec_suffix)
    if os.path.exists(nested_path):
        return nested_path
    flat_path = os.path.join(singlec_dir, sample + singlec_suffix)
    if os.path.exists(flat_path):
        return flat_path
    return None


def _compute_metrate_from_singlec(singlec_path, min_depth=1):
    """
    Genome-wide methylation ratio: sum(Met) / sum(Total) from singleC file.
    """
    total_c = 0.0
    met_c = 0.0
    for chunk in pd.read_csv(
        singlec_path,
        sep='\t',
        usecols=['Total', 'Met'],
        chunksize=200000,
    ):
        chunk = chunk.apply(pd.to_numeric, errors='coerce')
        chunk = chunk.dropna(subset=['Total', 'Met'])
        if min_depth > 1:
            chunk = chunk.loc[chunk['Total'] >= min_depth]
        if chunk.empty:
            continue
        total_c += float(chunk['Total'].sum())
        met_c += float(chunk['Met'].sum())
    if total_c <= 0:
        return np.nan
    return met_c / total_c


def _build_metrate_series(samples, singlec_dir, singlec_suffix, min_depth=1):
    """Build Met_Rate series aligned to sample index."""
    values = {}
    missing_singlec = []
    for sample in samples:
        singlec_path = _resolve_singlec_path(singlec_dir, sample, singlec_suffix)
        if singlec_path is None:
            missing_singlec.append(sample)
            values[sample] = np.nan
            continue
        values[sample] = _compute_metrate_from_singlec(singlec_path, min_depth=min_depth)
    return pd.Series(values, index=samples, dtype=float, name='Met_Rate'), missing_singlec


def _cal_ratio_from_stage1_csv(csv_path, gaps=1000):
    """Estimate contamination ratio from stage1 score CSV, aligned with cal_contamination_ratio.py."""
    df = pd.read_csv(csv_path)
    score_cols = [c for c in df.columns if 'score' in c.lower()]
    if 'C-score' in df.columns:
        score_col = 'C-score'
    elif score_cols:
        score_col = score_cols[0]
    else:
        raise ValueError(f'No score column found in {csv_path}')
    likelihood_1 = df[score_col].astype(np.float64).values
    likelihood_2 = 1.0 - likelihood_1
    best_sum, best_gap = -np.inf, 0.0
    for gap in np.linspace(0, 1, int(gaps)):
        total = np.nansum(np.log10(gap * likelihood_1 + (1.0 - gap) * likelihood_2))
        if total > best_sum:
            best_sum, best_gap = total, round(float(gap), 3)
    return best_gap


def _load_ratio_file(ratio_file):
    """Load contamination ratio file with columns: sample, ratio, loglik."""
    ratio_df = pd.read_csv(
        ratio_file,
        sep='\t',
        header=None,
        names=['sample_raw', 'r_contam', 'loglik'],
    )
    ratio_df['sample'] = ratio_df['sample_raw'].map(_normalize_sample_id)
    ratio_df['r_contam'] = pd.to_numeric(ratio_df['r_contam'], errors='coerce')
    ratio_df = ratio_df.dropna(subset=['sample', 'r_contam'])
    ratio_df = ratio_df.drop_duplicates(subset=['sample'], keep='last')
    ratio_df = ratio_df.set_index('sample')
    return ratio_df['r_contam']


def _build_ratio_series(samples, ratio_file=None, stage1_res_dir=None, ratio_grid_size=1000):
    """
    Build contamination ratio series for samples.
    Priority: ratio_file -> on-the-fly from stage1 score CSV.
    """
    if ratio_file and os.path.exists(ratio_file):
        ratio_series = _load_ratio_file(ratio_file)
        source = f'ratio_file:{ratio_file}'
    elif stage1_res_dir and os.path.exists(stage1_res_dir):
        values = {}
        for sample in samples:
            csv_path = os.path.join(stage1_res_dir, f'{sample}_id.csv')
            if not os.path.exists(csv_path):
                continue
            values[sample] = _cal_ratio_from_stage1_csv(csv_path, gaps=ratio_grid_size)
        ratio_series = pd.Series(values, dtype=float, name='r_contam')
        source = f'stage1_scores:{stage1_res_dir}'
    else:
        raise FileNotFoundError(
            'No contamination ratio source available. Provide ratio_file or valid stage1_res_dir.'
        )
    ratio_series = ratio_series.reindex(samples)
    missing = ratio_series[ratio_series.isna()].index.tolist()
    return ratio_series, source, missing


def _fit_adjustment_model(met_rate, r_contam):
    """Fit M_sample = alpha + beta * r_contam and return alpha, beta."""
    valid = met_rate.notna() & r_contam.notna()
    if valid.sum() < 2:
        raise ValueError('Not enough valid samples to fit Met_Rate adjustment model.')
    x = r_contam.loc[valid].astype(float).values
    y = met_rate.loc[valid].astype(float).values
    beta, alpha = np.polyfit(x, y, deg=1)
    return float(alpha), float(beta), int(valid.sum())


def target_methy_extraction(
    res_path,
    save_path,
    sample_info_path,
    sample_list_path=None,
    singlec_dir=None,
    singlec_suffix='_id_02.single5mC',
    singlec_min_depth=1,
    ratio_file=None,
    stage1_res_dir=None,
    ratio_grid_size=1000,
):
    """Aggregate target region methylation and build label.csv from sample_info."""
    os.makedirs(save_path, exist_ok=True)
    meth_list = []
    for f in os.listdir(res_path):
        if not f.endswith('.site_methylation.rate.xls') and not f.endswith('.xls'):
            continue
        feature_name = f.replace('.site_methylation.rate.xls', '').replace('.xls', '')
        df = pd.read_table(os.path.join(res_path, f), header=None, names=['file', 'total', 'meth', 'meth_rate'])
        df['sample'] = df['file'].str.replace('_' + feature_name + '.single5mC2', '', regex=False)
        df['feature'] = feature_name
        meth_list.append(df)
    if not meth_list:
        raise FileNotFoundError(f'No .site_methylation.rate.xls in {res_path}')
    all_meth = pd.concat(meth_list)
    new_df = all_meth.pivot_table(index='sample', columns='feature', values='meth_rate')
    new_df.index = new_df.index.astype(str).str.strip()
    if sample_list_path and os.path.exists(sample_list_path):
        samplelist = pd.read_csv(sample_list_path, header=None)
        keep = set(samplelist.iloc[:, 0].astype(str).str.strip())
        new_df = new_df.loc[new_df.index.isin(keep)]

    # Integrate sample-level methylation features for Stage2 TM model.
    if singlec_dir:
        met_rate, missing_singlec = _build_metrate_series(
            samples=new_df.index,
            singlec_dir=singlec_dir,
            singlec_suffix=singlec_suffix,
            min_depth=singlec_min_depth,
        )
        if missing_singlec:
            raise FileNotFoundError(
                f'Missing singleC files for {len(missing_singlec)} samples: '
                f'{missing_singlec[:5]}{"..." if len(missing_singlec) > 5 else ""}'
            )
        if met_rate.isna().any():
            na_samples = met_rate[met_rate.isna()].index.tolist()
            raise ValueError(
                f'Failed to compute Met_Rate for {len(na_samples)} samples: '
                f'{na_samples[:5]}{"..." if len(na_samples) > 5 else ""}'
            )
        new_df['Met_Rate'] = met_rate

        r_contam, ratio_source, missing_ratio = _build_ratio_series(
            samples=new_df.index,
            ratio_file=ratio_file,
            stage1_res_dir=stage1_res_dir,
            ratio_grid_size=ratio_grid_size,
        )
        if missing_ratio:
            raise ValueError(
                f'Missing contamination ratios for {len(missing_ratio)} samples: '
                f'{missing_ratio[:5]}{"..." if len(missing_ratio) > 5 else ""}'
            )

        alpha, beta, n_fit = _fit_adjustment_model(met_rate, r_contam)
        met_rate_adj = (met_rate - beta * r_contam).clip(lower=0.0, upper=1.0)
        new_df.insert(0, 'Met_Rate_adj', met_rate_adj)
        print(
            f'Met_Rate/Met_Rate_adj added: n={len(new_df)}, '
            f'ratio_source={ratio_source}, fit_n={n_fit}, alpha={alpha:.6f}, beta={beta:.6f}'
        )

    if sample_info_path.endswith('.xlsx') or sample_info_path.endswith('.xls'):
        label = pd.read_excel(sample_info_path)
    else:
        label = pd.read_csv(sample_info_path)
    if 'sample' not in label.columns and label.index.name is None:
        label = label.rename(columns={label.columns[0]: 'sample'})
    if 'sample' in label.columns:
        label = label.set_index('sample')
    label.index = label.index.astype(str).str.strip()
    overlap_cols = [c for c in label.columns if c in new_df.columns]
    if overlap_cols:
        print(f'Warning: dropping overlapping columns from sample_info: {overlap_cols}')
        label = label.drop(columns=overlap_cols)

    df = new_df.join(label, how='outer')
    df = df.dropna(how='all', subset=[c for c in new_df.columns if c in df.columns])
    label_cols = [c for c in label.columns if c in df.columns and c not in {'Met_Rate', 'Met_Rate_adj'}]
    # Keep label.csv focused on phenotype labels to avoid leaking numeric features.
    if any(c in PHENO_LABEL_COLS for c in label_cols):
        label_cols = [c for c in label_cols if c in PHENO_LABEL_COLS]
    if label_cols:
        df[label_cols].to_csv(os.path.join(save_path, 'label.csv'))
    df.to_csv(os.path.join(save_path, 'target_methy.csv'))
    print('target_methy shape:', df.shape)
    return df


def gwm_extraction(gwm_path, save_path, bin_bed_path=None):
    """Merge per-sample GWM bin files into one matrix; requires label.csv."""
    label = _read_label(save_path)
    samples = label.index.astype(str)
    if bin_bed_path and os.path.exists(bin_bed_path):
        gwbins = pd.read_table(bin_bed_path, header=None, names=['chrom', 'start', 'end'])
    else:
        gwbins = None
    res = None
    for f in os.listdir(gwm_path):
        if 'single5mC2' not in f:
            continue
        sample = f.replace('_gwm.single5mC2', '')
        df = pd.read_table(os.path.join(gwm_path, f), header=None,
                           names=['chrom', 'start', 'end', f'total_{sample}', f'methyed_{sample}', sample])
        df = df.drop(columns=[c for c in df.columns if c.startswith('total_') or c.startswith('methyed_')])
        if res is None:
            res = df.copy()
            if gwbins is not None:
                res = gwbins.merge(res, on=['chrom', 'start', 'end'], how='left')
        else:
            res = res.merge(df, on=['chrom', 'start', 'end'], how='left')
    if res is None:
        raise FileNotFoundError(f'No *_gwm.single5mC2 in {gwm_path}')
    res = res.loc[:, ~(res.isna().sum() == len(res))]
    chroms = ['chr' + str(i + 1) for i in range(22)] + ['chrX', 'chrY']
    res = res[res['chrom'].isin(chroms)]
    res_df = res.set_index(['chrom', 'start', 'end']).T
    res_df.index = res_df.index.astype(str)
    res_df = res_df.join(label)
    res_df = res_df.loc[samples.intersection(res_df.index)]
    res_df.to_csv(os.path.join(save_path, 'gwm.csv'))
    print('gwm shape:', res_df.shape)


def end_motif_extraction(save_path, bam_path, bam_suffix):
    """4-mer end motif frequency per sample (256 features)."""
    label = _read_label(save_path)
    samples = label.index.astype(str)
    bases = ['A', 'T', 'C', 'G']
    all_motif = [''.join(p) for p in product(bases, repeat=4)]
    rows = []
    for sample in tqdm(samples):
        bamfile = os.path.join(bam_path, sample, sample + bam_suffix)
        if not os.path.exists(bamfile):
            rows.append(pd.Series(0, index=all_motif))
            continue
        motifs = []
        with pysam.AlignmentFile(bamfile, 'rb') as bam:
            for r in bam:
                try:
                    xg = r.get_tag('XG')[3:-3]
                    motifs.append(xg[-4:] if r.is_reverse else xg[:4])
                except (KeyError, IndexError):
                    pass
        cnt = Counter(motifs)
        total = sum(cnt.values())
        row = pd.Series({m: cnt.get(m, 0) / total if total else 0 for m in all_motif})
        row.name = sample
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.join(label)
    df.to_csv(os.path.join(save_path, 'end_motif.csv'))
    print('end_motif shape:', df.shape)


def _extract_flen_one(sample, bam_path, bam_suffix):
    bamfile = os.path.join(bam_path, sample, sample + bam_suffix)
    flens = []
    try:
        with pysam.AlignmentFile(bamfile, 'rb') as bam:
            for r in bam:
                try:
                    xg = r.get_tag('XG')
                    flens.append(len(xg[3:-3]))
                except (KeyError, IndexError):
                    pass
    except Exception as e:
        print(f'Error {sample}: {e}')
    total = len(flens)
    if total == 0:
        return sample, {}
    return sample, {k: v / total for k, v in Counter(flens).items()}


def flen_extraction(save_path, bam_path, bam_suffix):
    """Fragment length distribution per sample."""
    label = _read_label(save_path)
    samples = label.index.astype(str).tolist()
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(_extract_flen_one, s, bam_path, bam_suffix): s for s in samples}
        result = {}
        for f in tqdm(as_completed(futures), total=len(futures)):
            sample, d = f.result()
            result[sample] = d
    flen_df = pd.DataFrame.from_dict(result, orient='index').fillna(0)
    flen_df = flen_df.join(label)
    flen_df = flen_df.loc[samples]
    flen_df.to_csv(os.path.join(save_path, 'flen.csv'))
    print('flen shape:', flen_df.shape)


def _load_bins(bin_bed):
    gwbins = pd.read_table(bin_bed, header=None, names=['chrom', 'start', 'end'])
    chroms = ['chr' + str(i + 1) for i in range(22)] + ['chrX', 'chrY']
    gwbins = gwbins[gwbins['chrom'].isin(chroms)]
    bins_by_chrom = {}
    for chrom in chroms:
        sub = gwbins[gwbins['chrom'] == chrom]
        bins_by_chrom[chrom] = {
            'starts': sub['start'].values,
            'ends': sub['end'].values,
            'indices': sub.index.values,
        }
    return bins_by_chrom, gwbins.index.tolist()


def _count_bins(bamfile, bins_by_chrom):
    counts = defaultdict(int)
    try:
        with pysam.AlignmentFile(bamfile, 'rb') as bam:
            for r in bam.fetch(until_eof=True):
                chrom = r.reference_name
                if chrom not in bins_by_chrom or r.reference_start is None:
                    continue
                mid = r.reference_start + (r.query_length or 0) // 2
                st = bins_by_chrom[chrom]['starts']
                en = bins_by_chrom[chrom]['ends']
                idx = bins_by_chrom[chrom]['indices']
                i = np.searchsorted(en, mid, side='left')
                if i < len(st) and st[i] <= mid < en[i]:
                    counts[idx[i]] += 1
    except Exception as e:
        print(f'BAM error {bamfile}: {e}')
    return counts


def _total_one(sample, bam_path, bam_suffix, bins_by_chrom):
    bamfile = os.path.join(bam_path, sample, sample + bam_suffix)
    return sample, _count_bins(bamfile, bins_by_chrom)


def total_extraction(save_path, bam_path, bam_suffix, bin_bed):
    """Read count per genome bin per sample (total)."""
    label = _read_label(save_path)
    samples = label.index.astype(str).tolist()
    bins_by_chrom, all_indices = _load_bins(bin_bed)
    n_bins = len(all_indices)
    total_counts = pd.DataFrame(0, index=all_indices, columns=samples, dtype=int)
    with ProcessPoolExecutor() as ex:
        futures = {ex.submit(_total_one, s, bam_path, bam_suffix, bins_by_chrom): s for s in samples}
        for future in tqdm(as_completed(futures), total=len(futures)):
            sample, counts = future.result()
            for idx, c in counts.items():
                total_counts.at[idx, sample] = c
    bin_df = pd.read_table(bin_bed, header=None, names=['chrom', 'start', 'end']).loc[all_indices]
    total_counts.index = [f"{r['chrom']}:{r['start']}-{r['end']}" for _, r in bin_df.iterrows()]
    total_counts = total_counts.T
    total_counts = total_counts.join(label).loc[samples]
    total_counts.to_csv(os.path.join(save_path, 'total.csv'))
    print('total shape:', total_counts.shape)


def cnv_extraction(cnv_path, save_path):
    """Merge CNA (e.g. Ginkgo) output; optional."""
    label = _read_label(save_path)
    cnv = pd.read_table(cnv_path)
    cnv = cnv.T
    cnv.columns = cnv.iloc[:3].apply(lambda col: '-'.join(map(str, col)), axis=0)
    cnv = cnv.drop(cnv.index[:3])
    cnv = cnv.astype(float) * 2
    cnv = cnv.join(label).loc[label.index]
    cnv.to_csv(os.path.join(save_path, 'cnv.csv'))
    print('cnv shape:', cnv.shape)


def main():
    ap = argparse.ArgumentParser(description='Extract 5 feature types (TRM, GWM, end_motif, flen, total; optional CNA).')
    ap.add_argument('--target_methy_path', help='Directory with target methylation .xls files')
    ap.add_argument('--gwm_path', help='Directory with *_gwm.single5mC2 files')
    ap.add_argument('--sample_info', help='Sample/label table (CSV or Excel) for building label.csv')
    ap.add_argument('--sample_list', help='Optional: list of sample IDs to keep')
    ap.add_argument('--singlec_dir', help='Directory containing per-sample singleC outputs for Met_Rate')
    ap.add_argument('--singlec_suffix', default='_id_02.single5mC', help='Suffix of singleC file per sample')
    ap.add_argument('--singlec_min_depth', type=int, default=1, help='Minimum Total depth when computing Met_Rate')
    ap.add_argument('--ratio_file', help='Optional contamination ratio file (sample\\tratio\\tloglik)')
    ap.add_argument('--stage1_res_dir', help='Fallback stage1 score CSV dir to compute contamination ratio')
    ap.add_argument('--ratio_grid_size', type=int, default=1000, help='Grid size for ratio MLE fallback')
    ap.add_argument('--bam_path', required=True, help='Base path to BAMs: bam_path/SAMPLE/SAMPLE{suffix}.bam')
    ap.add_argument('--bam_suffix', default='_id_02.bam')
    ap.add_argument('--save_path', required=True, help='Output directory; label.csv and feature CSVs written here')
    ap.add_argument('--bin_bed', help='BED for genome bins (for GWM merge and total)')
    ap.add_argument('--cnv_path', help='Optional CNA table path')
    ap.add_argument('--features', nargs='+', default=['target_methy', 'gwm', 'end_motif', 'flen', 'total'],
                    help='Which features to run')
    args = ap.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    if 'target_methy' in args.features and args.target_methy_path and args.sample_info:
        target_methy_extraction(
            args.target_methy_path,
            args.save_path,
            args.sample_info,
            args.sample_list,
            singlec_dir=args.singlec_dir,
            singlec_suffix=args.singlec_suffix,
            singlec_min_depth=args.singlec_min_depth,
            ratio_file=args.ratio_file,
            stage1_res_dir=args.stage1_res_dir,
            ratio_grid_size=args.ratio_grid_size,
        )
    if 'gwm' in args.features and args.gwm_path:
        gwm_extraction(args.gwm_path, args.save_path, args.bin_bed)
    if 'end_motif' in args.features:
        end_motif_extraction(args.save_path, args.bam_path, args.bam_suffix)
    if 'flen' in args.features:
        flen_extraction(args.save_path, args.bam_path, args.bam_suffix)
    if 'total' in args.features and args.bin_bed:
        total_extraction(args.save_path, args.bam_path, args.bam_suffix, args.bin_bed)
    if 'cnv' in args.features and args.cnv_path:
        cnv_extraction(args.cnv_path, args.save_path)


if __name__ == '__main__':
    main()
