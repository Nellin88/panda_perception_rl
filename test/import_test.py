import os
import sys

# Allow running this file directly from the repository root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def test_import():
    import panda_mujoco_gym
