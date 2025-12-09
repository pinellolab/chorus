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

# Load the model
enformer = hub.load(args['model_weights'])
# Get the actual model from the enformer object
model = enformer.model

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