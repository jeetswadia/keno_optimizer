# VERIFY GPU IS DETECTED — run this first:

print("=" * 50)
print("GPU DIAGNOSTIC")
print("=" * 50)

# TensorFlow check
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs detected: {gpus}")
    if gpus:
        print(f"  Name: {gpus[0]}")
        # Enable memory growth (important for RTX 2060 6GB)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("  ✅ TensorFlow WILL use GPU")
    else:
        print("  ❌ No GPU found by TensorFlow")
        print("  Install: pip install tensorflow==2.10.1")
except ImportError:
    print("TensorFlow not installed")

# PyTorch check
try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
       
        print("  ✅ PyTorch WILL use GPU")
except ImportError:
    print("PyTorch not installed")

# CuPy check (GPU NumPy)
try:
    import cupy as cp
    print(f"\nCuPy available: True")
    print(f"  Device: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
    print("  ✅ NumPy operations can use GPU")
except ImportError:
    print("\nCuPy not installed (optional, for GPU numpy)")