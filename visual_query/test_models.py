"""
Test that visual_query model classes can be instantiated with the config.
Does NOT require the Ego4D dataset - just verifies model loading.
"""
import sys, os, yaml
sys.path.insert(0, '..')
sys.path.insert(0, '../segment_anything')
sys.path.insert(0, '.')

import torch

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print("Testing REN model init (visual_query)...")
from models import REN

ren = REN(config['ren'])
print(f"  REN loaded, checkpoint epoch={ren.start_epoch}")
print(f"  Grid points shape: {ren.grid_points.shape}")

print("\nTesting CandidateSelector init...")
from models import CandidateSelector
selector = CandidateSelector()
print("  CandidateSelector OK")

print("\nAll visual_query model tests passed!")
