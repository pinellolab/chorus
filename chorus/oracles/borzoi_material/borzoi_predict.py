#!/usr/bin/env python
# Modified from borzoi_sed.py to predict expression from a sequence file
# Copyright 2022 Calico LLC (Modified)

from optparse import OptionParser
import json
import os
import tempfile

import h5py
import numpy as np
import pandas as pd
import pysam

from baskerville import dataset
from baskerville import seqnn

"""
borzoi_predict.py

Predict genomic features directly from DNA sequence file using the Borzoi model.
"""

def dna_1hot(seq):
    """Convert DNA sequence to one-hot encoding."""
    seq = seq.upper()
    seq_len = len(seq)
    seq_1hot = np.zeros((seq_len, 4), dtype='bool')
    
    for i in range(seq_len):
        if seq[i] == 'A':
            seq_1hot[i,0] = True
        elif seq[i] == 'C':
            seq_1hot[i,1] = True
        elif seq[i] == 'G':
            seq_1hot[i,2] = True
        elif seq[i] == 'T':
            seq_1hot[i,3] = True
    
    return seq_1hot

def main():
    usage = "usage: %prog [options] <params_file> <model_file> <sequence_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-o",
        dest="out_dir",
        default="pred_out",
        help="Output directory for predictions [Default: %default]",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "-u",
        dest="untransform_old",
        default=False,
        action="store_true",
        help="Untransform old models [Default: %default]",
    )
    parser.add_option(
        "--no_untransform",
        dest="no_untransform",
        default=False,
        action="store_true",
        help="Don't untransform predictions [Default: %default]",
    )
    parser.add_option(
        "--no_unclip",
        dest="no_unclip",
        default=False,
        action="store_true",
        help="Turn off unclip transform [Default: %default]",
    )
    (options, args) = parser.parse_args()

    if len(args) != 3:
        parser.error("Must provide parameters file, model file, and sequence file")
    else:
        params_file = args[0]
        model_file = args[1]
        sequence_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    options.shifts = [int(shift) for shift in options.shifts.split(",")]

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params["model"]
    params_train = params["train"]
    seq_len = params_model["seq_length"]

    if options.targets_file is None:
        parser.error("Must provide targets table to properly handle strands.")
    else:
        targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)

    # prep strand
    targets_strand_df = dataset.targets_prep_strand(targets_df)

    # set strand pairs (using new indexing)
    orig_new_index = dict(zip(targets_df.index, np.arange(targets_df.shape[0])))
    targets_strand_pair = np.array(
        [orig_new_index[ti] for ti in targets_df.strand_pair]
    )
    params_model["strand_pair"] = [targets_strand_pair]

    #################################################################
    # setup model

    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file)
    seqnn_model.build_slice(targets_df.index)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    model_stride = seqnn_model.model_strides[0]
    
    #################################################################
    # read sequence and predict

    # read sequence
    with open(sequence_file, 'r') as f:
        content = f.read().strip()
    
    # check if it's FASTA format
    if content.startswith('>'):
        # Simple FASTA parsing
        lines = content.split('\n')
        seq_id = lines[0][1:].strip()  # Remove > and any whitespace
        seq = ''.join(lines[1:]).replace('\n', '')
    else:
        # Assume it's just a raw sequence
        seq_id = "input_sequence"
        seq = content
    
    # Ensure sequence is proper length
    if len(seq) > seq_len:
        print(f"Warning: Input sequence length {len(seq)} exceeds model sequence length {seq_len}.")
        print(f"Truncating to {seq_len} bp.")
        # Take middle portion
        start = (len(seq) - seq_len) // 2
        seq = seq[start:start+seq_len]
    elif len(seq) < seq_len:
        print(f"Warning: Input sequence length {len(seq)} is shorter than model sequence length {seq_len}.")
        print(f"Padding with N's to reach {seq_len} bp.")
        # Pad with N's
        pad_left = (seq_len - len(seq)) // 2
        pad_right = seq_len - len(seq) - pad_left
        seq = 'N' * pad_left + seq + 'N' * pad_right
    
    # Convert sequence to one-hot encoding
    seq_1hot = dna_1hot(seq)
    seq_1hot = np.expand_dims(seq_1hot, axis=0)  # Add batch dimension
    
    # Get predictions
    preds = seqnn_model(seq_1hot)[0]
    
    # Untransform predictions
    if options.targets_file is not None:
        if not options.no_untransform:
            if options.untransform_old:
                preds = dataset.untransform_preds1(preds, targets_df, unclip=not options.no_unclip)
            else:
                preds = dataset.untransform_preds(preds, targets_df, unclip=not options.no_unclip)

    # Output prediction stats
    print(f"Prediction stats - min: {np.min(preds):.8f}, max: {np.max(preds):.8f}, mean: {np.mean(preds):.8f}")
    
    # Save predictions to HDF5
    h5_file = os.path.join(options.out_dir, 'predictions.h5')
    h5_out = h5py.File(h5_file, 'w')
    
    # Save predictions
    h5_out.create_dataset('preds', data=preds)
    
    # Save target information
    h5_out.create_dataset('target_ids', data=np.array(targets_df.identifier, 'S'))
    h5_out.create_dataset('target_labels', data=np.array(targets_df.description, 'S'))
    
    # Save sequence information
    h5_out.create_dataset('sequence', data=np.array([seq_id], 'S'))
    
    h5_out.close()
    
    print(f"Predictions saved to {h5_file}")


if __name__ == "__main__":
    main()
