# Shared utilities for DECENT-plus: BAM/reads processing and encoding.

import os
import gzip
import pickle
import pysam
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


def gdump(obj, filename):
    """Save object to gzipped pickle file."""
    with gzip.GzipFile(filename, 'wb') as f:
        pickle.dump(obj, f)


def gload(filename):
    """Load object from gzipped pickle file."""
    with gzip.GzipFile(filename, 'rb') as f:
        return pickle.load(f)


def file_name(directory):
    """Return (files, dirs) in the given directory."""
    for root, dirs, files in os.walk(directory):
        return files, dirs


def extract_reads(bam_path, reads_dir):
    """
    Extract reads from BAM (with XG/XM tags) into .reads files.
    Each line: query_name, ref_name, start, seq_132, methy_132.
    Skips reads with 'I' in CIGAR, 'N' in XG, or len(methylation) < 137.
    """
    bam = pysam.AlignmentFile(bam_path, 'rb')
    base = os.path.basename(bam_path)
    if base.endswith('.bam'):
        base = base[:-4]
    out_path = os.path.join(reads_dir, base + '.reads')
    with open(out_path, 'w') as out:
        for line in bam:
            if line.cigarstring and 'I' in line.cigarstring:
                continue
            try:
                read = line.get_tag('XG')
                methy = line.get_tag('XM')
            except KeyError:
                continue
            if 'N' in read:
                continue
            length = len(read)
            read = read[3 : length - 3]
            methylation = []
            for i in range(len(methy)):
                if methy[i] == 'X':
                    if i < len(methy) - 1:
                        methylation.append('1' if (read[i] == 'C' and read[i + 1] == 'G') else '0')
                    else:
                        methylation.append('1' if read[i] == 'C' else '0')
                else:
                    methylation.append('0')
            methylation = ''.join(methylation)
            if len(methylation) < 137:
                continue
            out.write(
                line.query_name + '\t' + line.reference_name + '\t'
                + str(line.reference_start) + '\t' + read[5:137] + '\t' + methylation[5:137] + '\n'
            )
    bam.close()
    return out_path


def reads_split(file_path):
    """Load .reads file; return (seq_list, methy_list) for lines with len(seq)==132."""
    seq, methy = [], []
    with open(file_path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 4 and len(parts[3]) == 132:
                seq.append(parts[3])
                methy.append(parts[4])
    return seq, methy


def lstm_seq(seq, methy):
    """Encode sequence to integer tensor; methylated C encoded as 4. methy: list of 0/1 strings or (N,132) float array."""
    encoder = LabelEncoder()
    encoder.fit(list('ACGT'))
    encoded = np.zeros((len(seq), len(seq[0])), dtype=np.int64)
    for i, s in enumerate(seq):
        encoded[i] = encoder.transform(list(s))
    lstmseq = torch.tensor(encoded, dtype=torch.int64)
    if isinstance(methy, np.ndarray):
        methy_arr = methy.astype(np.float32)
    else:
        methy_arr = np.array([[int(x) for x in line.strip()] for line in methy]).astype(np.float32)
    lstmseq[torch.tensor(methy_arr == 1)] = 4
    return lstmseq


def split_methy(seq, methy):
    """Convert seq and methy lists to LSTM-ready tensor."""
    split_methy_arr = np.array([[int(x) for x in line.strip()] for line in methy]).astype(np.float32)
    return lstm_seq(seq, split_methy_arr)


def conv_onehot(seq):
    """One-hot encode sequence (132, 5): A,T,C,G + methyl-C. Input shape (N, 132)."""
    onehot = torch.zeros(seq.size(0), 132, 5, dtype=torch.int64)
    onehot.scatter_(2, seq.unsqueeze(-1), 1)
    methy_mask = seq == 4
    onehot[methy_mask] = torch.tensor([0, 1, 0, 0, 1], dtype=torch.int64)
    return onehot
