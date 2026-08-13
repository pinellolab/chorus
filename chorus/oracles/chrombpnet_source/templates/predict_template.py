import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import json
import os
import sys

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script 
    args = json.load(inp)

sys.path.insert(0, str(args["BPNet_dir"])) # to get BPNet visible

def multinomial_nll(true_counts, logits):
        counts_per_example = tf.reduce_sum(true_counts, axis=-1)
        dist = tfp.distributions.Multinomial(total_count=counts_per_example,logits=logits)

        return (-tf.reduce_sum(dist.log_prob(true_counts)) /
                tf.cast(tf.shape(true_counts)[0], dtype=tf.float32))

# Configure device
device = args["device"]
if device:
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device.startswith('cuda:'):
        # TensorFlow selects its GPU through the mask, so `cuda:N` has to be
        # REMAPPED through any mask already in place rather than replacing it.
        # Overwriting it sent the process to physical GPU N, which under a
        # scheduler that granted e.g. CUDA_VISIBLE_DEVICES=4,5 is a device the
        # caller was never given. `cuda:N` means "the Nth device I can see".
        # See audits/2026-08-12_post_v0.7.2_audit.md F1.
        ordinal = device.split(':')[1]
        outer = os.environ.get('CUDA_VISIBLE_DEVICES')
        if outer:
            visible = [p for p in outer.split(',') if p != '']
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = visible[int(ordinal)]
            except (ValueError, IndexError):
                raise ValueError(
                    f"device='cuda:{ordinal}' but only {len(visible)} GPU(s) are "
                    f"visible to this process (CUDA_VISIBLE_DEVICES={outer!r}). "
                    f"cuda:N indexes the devices you were granted, not physical ones."
                )
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ordinal

# Read sequence from file
seq = args["sequence"]

# Load model
if args["is_CHIP"]:
    from BPNet.arch import BPNet
    
    # Open JSON for architecture
    with open(os.path.join(args["BPNet_dir"], "input_data.json")) as fopen:
        tasks_raw = json.load(fopen)
    tasks = {int(k): v for k, v in tasks_raw.items()}

    # Create model
    model = BPNet(tasks, {}, name_prefix="main")

    # Load weights
    model.load_weights(args["model_weights"])
else:
    model = tf.keras.models.load_model(
        args["model_weights"],
        compile=False,
        custom_objects={"multinomial_nll": multinomial_nll, "tf": tf}
    )

# Mapping dict
MAPPING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

if len(seq) > args["sequence_length"]:
    num_windows_stride_one = (len(seq) - args["sequence_length"] + 1)
    num_windows = (num_windows_stride_one + args["sequence_length"] - 1) // args["output_length"] + 1

    # Define seq_len and flag
    seq_len = (args["output_length"] * (num_windows - 1)) + args["sequence_length"]
    trimmed = False
else:
    seq_len = len(seq)
    trimmed = True

# One hot encoding
one_hot = np.zeros((seq_len, 4), dtype=np.float32)
for i, base in enumerate(seq.upper()):
    if base in MAPPING:
        one_hot[i, MAPPING[base]] = 1.0

# Add batch dimension
if trimmed:
    one_hot_batch = tf.constant(one_hot[np.newaxis], dtype=tf.float32)
else:
    # Compute windows of 2114 with a stride of 1000 to extend the prediction
    new_shape = (num_windows, args["sequence_length"], 4)
    stride_x, stride_y = one_hot.strides
    new_stride = (stride_x * args["output_length"], stride_x, stride_y)

    one_hot_batch = np.lib.stride_tricks.as_strided(one_hot, shape=new_shape, strides=new_stride)

# Extract predictions
if args["is_CHIP"]:
    # JASPAR models require bias profile and counts. Derive the shapes from the
    # model rather than hardcoding them: the count bias is declared (None, 2)
    # and is reduced by a logsumexp inside the model, so a hardcoded (N, 1) —
    # which Keras silently broadcasts instead of rejecting — makes that
    # logsumexp return log(1)=0 rather than log(2), shifting every predicted
    # log-count down by a constant 0.5885 (counts 1.8x too low at a peak, up to
    # 3x at a quiet site). See ChromBPNetOracle._zero_bias_inputs.
    batch_size = 1 if trimmed else num_windows
    bias_inputs = [
        np.zeros(
            [batch_size] + [d if d is not None else 1 for d in inp.shape[1:]],
            dtype="float32",
        )
        for inp in model.inputs[1:]
    ]
    result = model.predict_on_batch([one_hot_batch, *bias_inputs]) 
else:
    result = model.predict_on_batch(one_hot_batch)