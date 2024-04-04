import tensorflow as tf
from tensorflow.python.client import device_lib

# Check if GPU is available
print(tf.config.list_physical_devices('CPU'))
print(device_lib.list_local_devices())

if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is not available")
