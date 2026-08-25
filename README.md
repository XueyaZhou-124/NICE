# NICE

Source code for [NICE: A Two-Step Non-Invasive Framework for Embryo cfDNA Read Enrichment and Quality Assessment](https://doi.org/10.1002/advs.77327) (*Advanced Science*, 2026).

It employs **DECENT-plus** (a deep learning model) to distinguish between maternal contamination and embryo-derived reads, followed by a multi-dimensional feature extraction process to facilitate embryo quality assessment (probability of being a high-quality embryo).

---

## Overview

- **Input**: SECM BAM files with `XG`/`XM` methylation tags. Stage 1 training additionally uses scBS-seq from PB/Cumulus (maternal) and TE/ICM (embryonic).
- **Stage 1 (Purification)**: Train or apply DECENT-plus, score each read for maternal origin, and write a purified BAM of embryo-derived reads.
- **Stage 2 (Features & Classification)**: Extract target methylation, GWM, end motif, fragment length, genome-wide read counts, and optional CNA. Train or apply the NICE classifier.

---

## Environment

```bash
pip install -r requirements.txt
```

Also required on `$PATH`: `samtools`, `bedtools`, and `perl` (for single-C conversion).

Run Python modules from the repository root, and set `PYTHONPATH` to that root if needed:

```bash
export PYTHONPATH="$(pwd)"
```

---

## Data included in this repository

- `data/decent_plus_trained_30M.pth`: pretrained DECENT-plus weights.
- `data/TableS2_Target_region_feature/`: target-region annotation files used for target methylation features.


---

## Stage 1: DECENT-plus training and prediction

### 1.1 BAM preprocessing

Convert single-cell methylation BAMs (known cell types) into ID-labeled BAMs and `.reads` files:

```bash
python -m stage1_filtering.bam_preprocess id_bam \
  --original_bam_path <cell_type_sample>.bam \
  --new_bam_path <cell_type_sample>.id.bam

python -m stage1_filtering.bam_preprocess extract \
  --bam_path <cell_type_sample>.id.bam \
  --reads_dir <reads_dir>
```

### 1.2 Training

Default labels in `stage1_filtering/train.py`: Cumulus/PB = 0 (maternal), TE/ICM = 1 (embryo).

```bash
python -m stage1_filtering.train \
  --reads_dir <reads_dir> \
  --save_dir <checkpoint_dir> \
  --datasize 7500000 \
  --epochs 30 \
  --batch_size 128
```

### 1.3 SECM prediction and embryo-read filtering

```bash
python -m stage1_filtering.predict_and_filter \
  --bam_path <sample>.id.bam \
  --model_path data/decent_plus_trained_30M.pth \
  --reads_dir <temp_reads_dir> \
  --res_dir <output_csv_dir> \
  --threshold 0.2
```

- **CSV**: `<output_csv_dir>/<sample>.csv` with columns `header` and `C-score`. Higher C-score means more likely maternal.
- **BAM**: `<sample>_02.bam` keeps reads with C-score `<= 0.2` (purified embryo-derived reads).

Batch helpers:

```bash
bash scripts/run_stage1_predict_batch.sh <sample_list.txt> <BAM_ROOT> <MODEL_PATH> <RES_DIR> 0.2
bash scripts/run_stage1_predict_parallel.sh <sample_list.txt> <BAM_ROOT> <MODEL_PATH> <RES_DIR> 0.2 6
```

`sample_list.txt` is one sample ID per line. Each BAM is expected at `<BAM_ROOT>/<sample>/<sample>_id.bam`.

---

## Stage 2: Feature extraction and embryo assessment

### 2.1 BAM to single-C methylation

```bash
bash stage2_evaluation/run_bam_to_singleC.sh \
  <REF_FA> \
  <BAM_DIR> \
  <SAMPLE> \
  _id_02.bam \
  <OUT_SINGLEC_DIR> \
  stage2_evaluation/singleC_metLevel.hg19.pl
```

`REF_FA` should be an hg19 FASTA (plus lambda if used). Edit the reference path inside `singleC_metLevel.hg19.pl` to match your local genome FASTA before running.

Target-region methylation and genome-wide bin methylation (GWM) should be prepared from these single-C files, then aggregated in the next step. Region annotations live in `data/TableS2_Target_region_feature/`.

### 2.2 Feature matrix aggregation

```bash
python stage2_evaluation/feature_extraction.py \
  --target_methy_path <path_to_target_methylation_dir> \
  --gwm_path <path_to_gwm_dir> \
  --sample_info <sample_label.xlsx_or_csv> \
  --bam_path <bam_base_path> \
  --bam_suffix _id_02.bam \
  --save_path <feature_output_dir> \
  --bin_bed <hg19_5M.bed> \
  --cnv_path <cna.tsv>
```

One CSV is written per feature type:

1. **target_methy**: methylation at annotated target regions
2. **gwm**: genome-wide bin methylation
3. **end_motif**: 4-mer DNA end-motif frequencies
4. **flen**: fragment-length profiles
5. **total**: read counts in genome-wide bins
6. **cnv** (optional): copy-number variants

### 2.3 Embryo assessment

Train / cross-validate:

```bash
python stage2_evaluation/nice.py \
  --data_path <feature.csv> \
  --save_path <path_to_save_dir> \
  --k_features 10 \
  --valid_ratio 1 \
  --resample \
  --n_trials 30 \
  --thereshold 5 \
  --n_c 0.95 \
  --need_preprocess \
  --n_jobs 1
```

Predict with a saved model:

```bash
python stage2_evaluation/nice_predict.py \
  --model_path <final_model.joblib> \
  --data_path <feature.csv> \
  --save_path <predictions.csv>
```

---

## Utility scripts

- `scripts/cal_contamination_ratio.py`: estimate per-sample maternal contamination from Stage 1 score CSVs (MLE).

```bash
python scripts/cal_contamination_ratio.py \
  --res_dir <stage1_score_csv_dir> \
  --out_file <ratio.txt>
```

---

## Directory structure

```text
NICE/
├── stage1_filtering/
│   ├── bam_preprocess.py      # ID assignment and .reads extraction
│   ├── train.py               # DECENT-plus training
│   ├── predict_and_filter.py  # Per-read scoring and BAM filtering
│   ├── model.py
│   ├── data_loader.py
│   └── utils.py
├── stage2_evaluation/
│   ├── run_bam_to_singleC.sh
│   ├── singleC_metLevel.hg19.pl
│   ├── feature_extraction.py
│   ├── nice.py
│   ├── nice_predict.py
│   └── run_nice_best_config_5fold.sh
├── scripts/
│   ├── cal_contamination_ratio.py
│   ├── run_stage1_predict_batch.sh
│   └── run_stage1_predict_parallel.sh
├── data/
│   ├── decent_plus_trained_30M.pth
│   └── TableS2_Target_region_feature/
├── requirements.txt
└── LICENSE
```

---

## Citation

If you use this code, please cite:

Zhou, X., Ding, S., Zhang, Z., Shangguan, Q., Qiao, J., Zhou, P. & Chen, Y. NICE: A Two-Step Non-Invasive Framework for Embryo cfDNA Read Enrichment and Quality Assessment. *Advanced Science* (2026). https://doi.org/10.1002/advs.77327

```bibtex
@article{zhou2026nice,
  title   = {NICE: A Two-Step Non-Invasive Framework for Embryo cfDNA Read Enrichment and Quality Assessment},
  author  = {Zhou, Xueya and Ding, Shu and Zhang, Zhenyi and Shangguan, Qiaoling and Qiao, Jie and Zhou, Peijie and Chen, Yidong},
  journal = {Advanced Science},
  year    = {2026},
  pages   = {e77327},
  doi     = {10.1002/advs.77327},
  url     = {https://doi.org/10.1002/advs.77327}
}
```

---

**License**: [MIT License](LICENSE)

