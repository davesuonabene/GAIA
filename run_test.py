import sys
import os
import threading
import time
sys.path.append(os.path.abspath("."))
from core.mix import Mix
from audio.engine import Engine
import numpy as np

def run_test():
    print("Initializing Engine...")
    engine = Engine(44100)
    audio = np.random.randn(2, 44100 * 2) # 2 seconds
    
    print("Separating Stems...")
    stems = engine.separate_stems(audio, 44100)
    print("Stems returned:", stems.keys())
    
    # Simulate exactly what cli.py does in the callback
    print("Testing Pre-Mix callback logic...")
    min_len = min([s.shape[1] for s in stems.values()])
    print("Min stem length:", min_len)
    
    pre_mixed = np.zeros((audio.shape[0], min_len))
    gains = {'vocals': 1.0, 'drums': 1.0, 'bass': 1.0, 'other': 1.0}
    
    for name, stem_audio in stems.items():
        pre_mixed += stem_audio[:, :min_len] * gains[name]
        
    print("Pre-mixed max value:", np.max(pre_mixed))
    print("Pre-mixed shape:", pre_mixed.shape)
    
run_test()
