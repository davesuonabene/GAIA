import numpy as np
import pedalboard
import time

def test_pedalboard_reset():
    samplerate = 44100
    comp = pedalboard.Compressor(threshold_db=-20, ratio=10)
    
    # Generate some loud audio
    audio = np.ones((1, 1024), dtype=np.float32) * 0.5
    
    # Process twice with .process()
    out1 = comp.process(audio, samplerate)
    out2 = comp.process(audio, samplerate)
    
    # If it resets, out1 and out2 should be identical (since input is identical and starting state is same)
    # If it DOESN'T reset, out2 should be more compressed than out1 (because the envelope follower has already reacted)
    
    print(f"out1 mean: {np.mean(np.abs(out1))}")
    print(f"out2 mean: {np.mean(np.abs(out2))}")
    
    if np.allclose(out1, out2):
        print("Pedalboard.process() resets state by default!")
    else:
        print("Pedalboard.process() maintains state by default!")

if __name__ == "__main__":
    try:
        test_pedalboard_reset()
    except Exception as e:
        print(f"Error: {e}")
