import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.audio_module import TransientShaperModule, SaturationModule, CompressorModule
from core.parameter import Parameter
from audio.crossover import Crossover

def test_fx():
    print("Testing TransientShaperModule...")
    ts = TransientShaperModule()
    ts.parameters["Attack Boost"].current_value = 10.0
    ts.parameters["Sustain Boost"].current_value = 10.0
    
    sample_rate = 44100
    # Create 1 second of stereo noise
    audio = np.random.uniform(-0.5, 0.5, (2, sample_rate)).astype(np.float32)
    
    try:
        out_ts = ts.process(audio, sample_rate)
        print("TransientShaperModule passed. Output shape:", out_ts.shape)
    except Exception as e:
        print("TransientShaperModule failed:", e)

    print("\nTesting SaturationModule...")
    sat = SaturationModule()
    sat.parameters["Drive"].current_value = 20.0
    sat.parameters["Mix"].current_value = 50.0
    
    try:
        out_sat = sat.process(audio, sample_rate)
        print("SaturationModule passed. Output shape:", out_sat.shape)
    except Exception as e:
        print("SaturationModule failed:", e)

    print("\nTesting Crossover...")
    co = Crossover([1000.0, 5000.0], fs=sample_rate)
    
    try:
        bands = co.split_chunk(audio)
        print(f"Crossover split passed. Generated {len(bands)} bands.")
        
        summed = co.sum_bands(bands)
        print("Crossover sum passed. Summed shape:", summed.shape)
        
        # Test perfect reconstruction (Linkwitz-Riley is flat in magnitude, but phase shifted)
        # We can't simply subtract because it's IIR and phase is shifted, 
        # but we ensure it doesn't crash and outputs look sane.
        max_val = np.max(np.abs(summed))
        print(f"Crossover output max val: {max_val} (input max roughly 0.5)")
        
    except Exception as e:
        print("Crossover failed:", e)
        
    print("\nAll tests finished.")

if __name__ == "__main__":
    test_fx()
