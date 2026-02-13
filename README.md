# SECM-cfDNA Embryo Origin Discrimination and Quality Assessment Pipeline

Source code for *NICE: A Two-Step Non-Invasive Framework for Embryo cfDNA Read Enrichment and Quality Assessment*

 It employs **DECENT-plus** (a deep learning model) to distinguish between maternal contamination and embryo-derived reads, followed by a multi-dimensional feature extraction process to facilitate embryo quality assessment (Probability of being a "High-quality Embryo").

---

## 📋 Overview

* **Input**: SECM BAM files (containing `XG/XM` methylation tags) and scBS-seq files from PB/Cumulus(maternal source) & TE/ICM(embyoric source) (for stage1 *Decent-plus* training).
* **Stage 1 (Purification)**: Train DECENT-plus  predict maternal probability per read, Filter embryo-derived reads based on a threshold, Generate "purified" BAM.
* **Stage 2 (Feature Extraction & Classification)**: Extract 6 categories of features (TM, TRN, GWM, EM, FLEN, CNA). Generate one CSV per feature type (one row per sample). And downstream machine learning for embryo quality assessment.

---

## 💻 Environment

```bash
pip install -r requirements.txt

```

**External Dependencies**:

* `samtools`, `bedtools` must be installed in your `$PATH`.


---

## 🔬 Stage 1: DECENT-plus Training & Prediction

### 1.1 BAM Preprocessing (Preparing for Training)

Convert **Single-cell Methylation BAMs** (known cell types) into ID-labeled BAMs and `.reads` files:

```bash
# Assign numerical IDs to every read in the BAM (for tracking)
python -m stage1_decent.bam_preprocess id_bam \
  --original_bam_path <cell_type_sample>.bam \
  --new_bam_path <cell_type_sample>.id.bam

# Extract .reads files (Sequence + Methylation status for 132bp)
python -m stage1_decent.bam_preprocess extract \
  --bam_path <cell_type_sample>.id.bam \
  --reads_dir <reads_dir>

```

### 1.2 Training the Model

Configure cell types in `stage1_decent/train.py` (Default: Cumulus/PB = 0 [Maternal], TE/ICM = 1 [Embryo]):

```bash
python -m stage1_decent.train \
  --reads_dir <reads_dir> \
  --save_dir <checkpoint_dir> \
  [--datasize 7500000] [--epochs 30] [--batch_size 128]

```

### 1.3 SECM Prediction & Embryo Read Filtering

Run prediction on SECM `.id.bam` files:

```bash
python -m stage1_decent.predict_and_filter \
  --bam_path <sample>.id.bam \
  --model_path <path/to/model_epoch_15.pth> \ # trained decent-plus
  --reads_dir <temp_reads_dir> \
  --res_dir <output_csv_dir> \ # probability of maternal source per read
  --threshold 0.2 # filter threshold for reads filteration

```

* **CSV Output**: `res_dir/<sample>.csv` (Columns: `header`, `C-score`). Reads with **C-score** are classified as Maternal-derived.
* **BAM Output**: `<sample>_02.bam` (Contains only reads with C-score <= 0.2, serve as purified reads from embyro).

---

## 📊 Stage 2: Feature Extraction

### 2.1 Low-level Processing (BAM to Methylation)

* **BAM to singleC**: Use `stage2_features/run_bam_to_singleC.sh`.
* **Target Region Methylation (TRM)**: Use `run_target_methylation.sh`.
* **Genome-wide Bin Methylation (GWM)**: Use `run_gwm.sh`.

### 2.2 Feature Matrix Aggregation

Aggregate all extracted data into structured CSVs using the master extraction script:

```bash
python stage2_features/feature_extraction.py \
  --target_methy_path <path_to_target_methylation_dir> \
  --gwm_path <path_to_gwm_dir> \
  --sample_info <sample_label.xlsx_or_csv> \
  --bam_path <bam_base_path> \
  --bam_suffix _id_02.bam \
  --save_path <feature_output_dir> \
  [--bin_bed <hg19_5M.bed>] [--cna_path <cna.tsv>]

```

**Extracted Features (One CSV per type):**

1. **Target Methylation (TRM)**: Methylation levels at specific genomic regions.
2. **GWM**: Genome-wide bin-based methylation.
3. **End Motif**: Frequency of 4-mer DNA end motifs.
4. **Flen**: Fragment length distribution profiles.
5. **Total**: Total reads number on genome-wide bin.
6. **CNV**: Copy Number Variations profiles.

### 2.3 Embyro Assement

Clssify high-quality embyro based on extracted features:

```bash
python stage2_evaluation/nice.py \
 --data_path target_methy.csv \ # extracted feature
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

---

## 🛠 Utility Scripts

* **`scripts/cal_contamination_ratio.py`**: Estimates the global maternal contamination ratio for a sample using Maximum Likelihood Estimation (MLE) based on Stage 1 scores.

---

## 📂 Directory Structure

```text
NICE/
├── stage1_filtering/           # Training & Prediction (DL)
│   ├── bam_preprocess.py    # ID assignment & read extraction
│   ├── train.py             # Model training
│   ├── predict_and_filter.py# Filtering embryo reads
│   └── model.py             # Neural network architecture
├── stage2_evaluation/         # Feature Engineering
│   ├── run_bam_to_singleC.sh
│   ├── run_gwm.sh
│   └── feature_extraction.py  # Aggregates 5 feature types into CSVs
└── scripts/
    └── cal_contamination_ratio.py # Contamination estimation

```

---

## 📝 Notes & Citation

* **C-score**: Represents the probability of a read being **Maternal**. Therefore, `1 - C-score` is the Embryo probability.

**License**: [MIT License]

Would you like me to help you generate a `requirements.txt` based on these modules?