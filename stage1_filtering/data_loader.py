"""Training data preparation: load reads by cell-type suffix and build tensors."""

import os
import random
import torch
import numpy as np
from utils import file_name, reads_split, split_methy, conv_onehot


class Example:
    """Prepare per-cell-type reads and encode to one-hot tensors for training."""

    def __init__(self, cell_type_suffix_dict, reads_dir, device="cpu", datasize=7500000, seed=42, reads_len=132):
        self.cell_type_suffix_dict = cell_type_suffix_dict
        self.reads_dir = reads_dir
        self.device = device
        self.datasize = datasize
        self.seed = seed
        self.reads_len = reads_len

    def data_prepare(self, suffix):
        """Load seq and methy from .reads files whose name contains suffix."""
        files, _ = file_name(self.reads_dir)
        seq, methy = [], []
        for f in files:
            if str(suffix) not in f:
                continue
            path = os.path.join(self.reads_dir, f)
            s, m = reads_split(path)
            seq.extend(s)
            methy.extend(m)
        return seq, methy

    def random_sample(self, seq, methy):
        """Random sample up to datasize."""
        random.seed(self.seed)
        n = min(len(seq), self.datasize)
        idx = random.sample(range(len(seq)), n)
        return [seq[i] for i in idx], [methy[i] for i in idx]

    def data_to_dict(self):
        """Return dict: cell_type -> [seq, methy, seq_lstm, seq_one_hot]."""
        out = {}
        for cell_type, suffix in self.cell_type_suffix_dict.items():
            seq, methy = self.data_prepare(suffix)
            seq, methy = self.random_sample(seq, methy)
            seq_lstm = split_methy(seq, methy)
            seq_one_hot = conv_onehot(seq_lstm)
            out[cell_type] = [seq, methy, seq_lstm, seq_one_hot]
        return out


def data_prepare(cell_type_dict, cell_type_suffix_dict, reads_dir, datasize=7500000, reads_len=132):
    """Build stacked data and labels for training."""
    ex = Example(
        cell_type_suffix_dict=cell_type_suffix_dict,
        reads_dir=reads_dir,
        device="cpu",
        datasize=datasize,
        reads_len=reads_len,
    )
    sample_dict = ex.data_to_dict()
    data = torch.vstack([sample_dict[k][3] for k in cell_type_dict.keys()])
    label = np.array(
        [[cell_type_dict[k]] * sample_dict[k][3].size(0) for k in cell_type_dict.keys()]
    ).flatten()
    return data, label
