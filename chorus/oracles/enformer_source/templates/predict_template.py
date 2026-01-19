import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import json 
#import pandas as pd
from chorus.oracles.enformer_source.enformer_metadata import get_metadata

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script 
    args = json.load(inp)


# Configure device
device = args['device']
if device:
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device.startswith('cuda:'):
        gpu_id = device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

# Read sequence from file
seq = args['sequence']


# Load model (cached in TFHub)
# Enformer model has a specific structure - we need to get the model attribute
enformer = hub.load(args['model_weights'])
model = enformer.model

# One-hot encode
mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
one_hot = np.zeros((len(seq), 4), dtype=np.float32)

for i, base in enumerate(seq.upper()):
    if base in mapping:
        one_hot[i, mapping[base]] = 1.0

# Add batch dimension
one_hot_batch = tf.constant(one_hot[np.newaxis], dtype=tf.float32)

# Run prediction - Use predict_on_batch method
predictions = model.predict_on_batch(one_hot_batch)
# Extract human predictions (Enformer outputs both human and mouse)
human_predictions = predictions['human'][0].numpy()

meta = get_metadata()
track_indices = meta.id2index(args['assay_ids'])
if any(map(lambda x: x is None, track_indices)):
    raise Exception(f"Some assay IDs not found in metadata: {args['assay_ids']}")
# Extract predictions for selected tracks
selected_predictions = human_predictions[:, track_indices]
result = selected_predictions.tolist()