import os
import sys

# Make the pipeline root importable so tests can `import eval.metrics`,
# `import training.rewards.cosine_length`, etc. when pytest is run from the
# repo root or from pipeline/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
