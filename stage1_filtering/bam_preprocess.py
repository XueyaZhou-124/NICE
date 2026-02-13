"""
Stage1 preprocessing: (1) Assign numeric IDs to BAM read names; (2) Extract reads to .reads files.
Run from project root: python -m stage1_decent.bam_preprocess [step] ...
Or from stage1_decent: python bam_preprocess.py ...
"""

import os
import sys
import argparse
import pysam

try:
    from .utils import extract_reads
except ImportError:
    from utils import extract_reads


def id_bam(original_bam_path, new_bam_path):
    """Rewrite BAM so each read has query_name = zero-padded row index (e.g. 00000000)."""
    original_bam = pysam.AlignmentFile(original_bam_path, 'rb')
    new_bam = pysam.AlignmentFile(new_bam_path, 'wb', header=original_bam.header)
    for count, read in enumerate(original_bam):
        read.query_name = f'{count:08d}'
        new_bam.write(read)
    original_bam.close()
    new_bam.close()


def main_id_bam():
    parser = argparse.ArgumentParser(description='Assign numeric IDs to BAM read names.')
    parser.add_argument('--original_bam_path', required=True, help='Input BAM')
    parser.add_argument('--new_bam_path', required=True, help='Output BAM')
    args = parser.parse_args()
    id_bam(args.original_bam_path, args.new_bam_path)


def main_extract_reads():
    parser = argparse.ArgumentParser(description='Extract reads from BAM to .reads files.')
    parser.add_argument('--bam_path', required=True, help='Path to .id.bam file')
    parser.add_argument('--reads_dir', required=True, help='Directory to write .reads files')
    args = parser.parse_args()
    extract_reads(args.bam_path, args.reads_dir)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'id_bam':
        sys.argv.pop(1)
        main_id_bam()
    elif len(sys.argv) > 1 and sys.argv[1] == 'extract':
        sys.argv.pop(1)
        main_extract_reads()
    else:
        print('Usage: python bam_preprocess.py id_bam --original_bam_path <in.bam> --new_bam_path <out.id.bam>')
        print('       python bam_preprocess.py extract --bam_path <sample.id.bam> --reads_dir <dir>')
        sys.exit(1)
