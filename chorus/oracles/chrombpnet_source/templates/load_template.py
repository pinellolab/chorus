import tensorflow as tf
import tensorflow_probability as tfp
import json
import os
import sys

from tensorflow.keras.utils import get_custom_objects

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
    if device=='cpu':
        # Force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Forcing CPU usage")
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
            visible = [x for x in outer.split(',') if x != '']
            try:
                gpu_id = visible[int(ordinal)]
            except (ValueError, IndexError):
                raise ValueError(
                    f"device='cuda:{ordinal}' but only {len(visible)} GPU(s) are "
                    f"visible to this process (CUDA_VISIBLE_DEVICES={outer!r}). "
                    f"cuda:N indexes the devices you were granted, not physical ones."
                )
        else:
            gpu_id = ordinal
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"Using GPU {gpu_id} (requested {device})")
    elif device in ['cuda', 'gpu']:
        # Use default GPU (don't change CUDA_VISIBLE_DEVICES)
        print("Using default GPU")

else:
    # Auto-detect - TensorFlow will use GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Auto-detected {len(gpus)} GPU(s), using first available")
    else:
        print("No GPU detected, using CPU")

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

# Get device info
if device == 'cpu' or not tf.config.list_physical_devices('GPU'):
    actual_device = 'CPU'
else:
    actual_device = f'GPU ({len(tf.config.list_physical_devices("GPU"))} available)'

# Get model info (we can't pickle the model itself)
result = {
    'loaded': True,
    'model_class': str(type(model)),
    'has_predict': hasattr(model, 'predict_on_batch'),
    'description': 'ChromBPNet model loaded successfully',
    'device': actual_device
}