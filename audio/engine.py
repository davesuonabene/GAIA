import numpy as np
import pedalboard
from typing import List
import sys
import os

try:
    from .crossover import Crossover
    from ..core.mix import Mix
    from ..core.audio_module import CompressorModule
except (ImportError, ValueError):
    try:
        from audio.crossover import Crossover
    except ImportError:
        from crossover import Crossover
    from core.mix import Mix
    from core.audio_module import CompressorModule

class Engine:
    """The core audio engine that maps DNA to DSP."""
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def process(self, input_audio: np.ndarray, mix_dna: Mix) -> np.ndarray:
        """Processes the input audio using the provided Mix DNA."""
        # 1. Split audio into bands using the Mix's crossovers
        crossover = Crossover(mix_dna.crossovers, self.sample_rate)
        bands_audio = crossover.split(input_audio)
        
        processed_bands = []
        
        # 2. Process each band with its corresponding module chain
        for i, band_dna in enumerate(mix_dna.bands):
            band_audio = bands_audio[i]
            
            # Map Band DNA to a Pedalboard chain
            chain = []
            for module in band_dna.modules:
                pb_module = self._map_module(module)
                if pb_module:
                    chain.append(pb_module)
            
            # Create a Pedalboard and process the band audio
            if chain:
                board = pedalboard.Pedalboard(chain)
                processed_band = board.process(band_audio, self.sample_rate)
                processed_bands.append(processed_band)
            else:
                processed_bands.append(band_audio)
                
        # 3. Sum the processed bands back together
        return crossover.sum_bands(processed_bands)

    def _map_module(self, module):
        """Maps a custom AudioModule DNA object to a Pedalboard effect."""
        if isinstance(module, CompressorModule):
            params = module.parameters
            # Mapping our DNA parameters to Pedalboard Compressor parameters
            return pedalboard.Compressor(
                threshold_db=params["Threshold"].current_value,
                ratio=params["Ratio"].current_value,
                attack_ms=params["Attack"].current_value,
                release_ms=params["Release"].current_value,
                # Pedalboard doesn't have a direct "makeup gain" in its basic Compressor,
                # but it does have output_gain_db in some versions or we can use Gain module.
                # For simplicity, we'll focus on these primary ones.
            )
        return None

if __name__ == "__main__":
    # --- ENGINE SMOKE TEST ---
    from core.mix import Mix
    from core.audio_module import CompressorModule
    
    # 1. Setup a simple mix with a compressor in the low band
    mix = Mix(crossovers=[150.0])
    low_band = mix.bands[0]
    comp = CompressorModule()
    comp.parameters["Threshold"].current_value = -20.0 # Heavy compression
    comp.parameters["Ratio"].current_value = 4.0
    low_band.modules.append(comp)
    
    # 2. Generate test signal (1s noise)
    fs = 44100
    input_signal = np.random.uniform(-0.5, 0.5, fs)
    
    # 3. Process through Engine
    engine = Engine(sample_rate=fs)
    output_signal = engine.process(input_signal, mix)
    
    print("=== ENGINE SMOKE TEST ===")
    print(f"Input Signal RMS: {np.sqrt(np.mean(input_signal**2)):.6f}")
    print(f"Output Signal RMS: {np.sqrt(np.mean(output_signal**2)):.6f}")
    print(f"Bands processed: {len(mix.bands)}")
    print("Engine process completed successfully.")
