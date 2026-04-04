import sys
import os
import numpy as np
sys.path.append(os.path.abspath("."))
from core.mix import Mix
from core.band import Band
from core.audio_module import CompressorModule, TransientShaperModule
from audio.engine import Engine

def test_engine_init():
    print("Testing Engine Initialization...")
    engine = Engine(44100)
    mix = Mix(crossovers=[100, 500, 2000])
    
    # Add some modules to a band
    mix.bands[0].modules.append(CompressorModule())
    mix.bands[1].modules.append(TransientShaperModule())
    
    print("Initial Mix Compilation...")
    engine.update_mix(mix)
    
    print("Processing chunk...")
    audio = np.random.randn(2, 512).astype(np.float32)
    output = engine.process_chunk(audio)
    
    assert output.shape == audio.shape
    print("Processing chunk PASSED!")
    
    print("Testing In-Place Update (No topology change)...")
    mix.bands[0].modules[0].parameters["Threshold"].current_value = -20.0
    engine.update_mix(mix)
    print("In-Place Update PASSED!")
    
    print("Testing Topology Change...")
    mix.bands[0].modules.append(CompressorModule())
    engine.update_mix(mix)
    print("Topology Change PASSED!")
    
    print("All basic engine tests PASSED!")

if __name__ == "__main__":
    try:
        test_engine_init()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
