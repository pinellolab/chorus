import tensorflow as tf
import tensorflow_hub as hub
import os
import json

with open("__ARGS_FILE_NAME__") as inp:  # to be formatted by calling script
    args = json.load(inp)

# Set TFHub progress tracking
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "1"

# Configure device
device = args['device']
if device:
    if device == 'cpu':
        # Force CPU usage
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Forcing CPU usage")
    elif device.startswith('cuda:'):
        # Use specific GPU
        gpu_id = device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"Using GPU {{gpu_id}}")
    elif device in ['cuda', 'gpu']:
        # Use default GPU (don't change CUDA_VISIBLE_DEVICES)
        print("Using default GPU")
else:
    # Auto-detect - TensorFlow will use GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Auto-detected {{len(gpus)}} GPU(s), using first available")
    else:
        print("No GPU detected, using CPU")


# Load the model. Chorus prefers the HF mirror at
# `lucapinello/chorus-enformer` (mirror of the original DeepMind
# SavedModel) for resilience — the original TFHub URL now redirects
# through Kaggle and the migration trail has caused breakage. We fall
# back to the original TFHub URL if the HF mirror is unreachable.
def _load_with_recovery(weights: str):
    """Load via huggingface_hub when weights is a HF repo id, else via
    tensorflow_hub. On any failure for the HF path, fall back to the
    original TFHub URL the load template was designed for. On TFHub
    cache corruption (incomplete download from a previous session —
    missing saved_model.pb), clear the bad cache directory and retry."""
    import re, shutil

    # HF repo path: not a URL, looks like "user/repo-name".
    if "/" in weights and not weights.startswith("http"):
        try:
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                repo_id=weights,
                repo_type="model",
                allow_patterns=["saved_model.pb", "variables/*"],
            )
            print(f"Loading Enformer SavedModel from HF mirror at {{local_dir}}")
            return tf.saved_model.load(local_dir)
        except Exception as exc:
            print(f"HF mirror load failed ({{exc}}); falling back to TFHub")
            weights = "https://tfhub.dev/deepmind/enformer/1"

    try:
        return hub.load(weights)
    except Exception as exc:
        msg = str(exc)
        if "saved_model.pb" not in msg:
            raise
        m = re.search(r"'([^']*tfhub_modules[^']+)'", msg)
        if not m:
            raise
        bad_dir = m.group(1)
        if os.path.isdir(bad_dir):
            print(f"Clearing corrupt tfhub cache at {{bad_dir}}")
            shutil.rmtree(bad_dir, ignore_errors=True)
        return hub.load(weights)


loaded = _load_with_recovery(args['model_weights'])

# The HF SavedModel is the *underlying* tf.saved_model.load output, while
# the TFHub `hub.load` wraps it in an "enformer" object with a `.model`
# attribute. Detect which we got and unwrap.
if hasattr(loaded, "model"):
    model = loaded.model
else:
    model = loaded

# Get device info
if device == 'cpu' or not tf.config.list_physical_devices('GPU'):
    actual_device = 'CPU'
else:
    actual_device = f'GPU ({{len(tf.config.list_physical_devices("GPU"))}} available)'

# Get model info (we can't pickle the model itself)
result = {
    'loaded': True,
    'model_class': str(type(model)),
    'has_predict': hasattr(model, 'predict_on_batch'),
    'description': 'Enformer model loaded successfully',
    'device': actual_device
}
