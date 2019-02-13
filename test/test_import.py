# Create a new test file: test_detailed.py
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

print("Basic imports successful")

# Test each import step by step
try:
    print("Testing dlplay.utils.types...")
    from dlplay.utils.types import ArrayLike
    print("✓ dlplay.utils.types imported successfully")
except Exception as e:
    print(f"✗ dlplay.utils.types failed: {e}")

try:
    print("Testing dlplay.core.tensor_backend...")
    from dlplay.core.tensor_backend import TensorBackend
    print("✓ dlplay.core.tensor_backend imported successfully")
except Exception as e:
    print(f"✗ dlplay.core.tensor_backend failed: {e}")

try:
    print("Testing dlplay.viz.plotting...")
    from dlplay.viz.plotting import plot_grad_descent_paths
    print("✓ dlplay.viz.plotting imported successfully")
except Exception as e:
    print(f"✗ dlplay.viz.plotting failed: {e}")

try:
    print("Testing dlplay.optimization.optimization...")
    from dlplay.optimization.optimization import ObjectiveFunction
    print("✓ dlplay.optimization.optimization imported successfully")
except Exception as e:
    print(f"✗ dlplay.optimization.optimization failed: {e}")