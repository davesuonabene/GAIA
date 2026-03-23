import numpy as np
import pedalboard
from typing import List
import sys
import os

try:
    from .crossover import Crossover
    from ..core.mix import Mix
    from ..core.audio_module import (
        CompressorModule, ExpanderModule, ClipperModule, 
        LimiterModule, ConvolutionModule, SaturationModule, TransientShaperModule
    )
    from ..core.band import Band
except (ImportError, ValueError):
    try:
        from audio.crossover import Crossover
    except ImportError:
        from crossover import Crossover
    from core.mix import Mix
    from core.audio_module import (
        CompressorModule, ExpanderModule, ClipperModule, 
        LimiterModule, ConvolutionModule, SaturationModule, TransientShaperModule
    )
    from core.band import Band

class Engine:
    """The core audio engine that maps DNA to DSP."""
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.safety_limiter = None
        if pedalboard:
            self.safety_limiter = pedalboard.Limiter(threshold_db=-0.1)

    def _process_band(self, audio: np.ndarray, band_dna: Band) -> np.ndarray:
        """Process a single band's audio through its module chain and gain."""
        current_audio = audio
        for module in band_dna.modules:
            # Call module.process directly
            current_audio = module.process(current_audio, self.sample_rate)
        
        # Apply Band Gain
        linear_gain = 10 ** (band_dna.gain.current_value / 20.0)
        return current_audio * linear_gain

    def process(self, input_audio: np.ndarray, mix_dna: Mix) -> np.ndarray:
        """Processes the input audio using the provided Mix DNA."""
        # 1. Process through PRE band
        current_audio = self._process_band(input_audio, mix_dna.pre_band)

        # 2. Split audio into bands using the Mix's crossovers
        crossover = Crossover(mix_dna.crossovers, self.sample_rate)
        bands_audio = crossover.split(current_audio)
        
        # 3. Check for Solo state
        solo_active = any(band.is_soloed for band in mix_dna.bands)
        
        processed_bands = []
        
        # 4. Process each band with its corresponding module chain
        for i, band_dna in enumerate(mix_dna.bands):
            band_audio = bands_audio[i]
            
            # Solo/Mute logic (applied only to parallel frequency bands)
            if solo_active:
                if not band_dna.is_soloed:
                    processed_bands.append(np.zeros_like(band_audio))
                    continue
            elif band_dna.is_muted:
                processed_bands.append(np.zeros_like(band_audio))
                continue

            processed_bands.append(self._process_band(band_audio, band_dna))
                
        # 5. Sum the processed bands back together
        summed_audio = crossover.sum_bands(processed_bands)
        
        # 6. Process through POST band
        final_audio = self._process_band(summed_audio, mix_dna.post_band)
        
        # 7. Final Limiter to prevent clipping (safety)
        if self.safety_limiter:
            return self.safety_limiter.process(final_audio, self.sample_rate)
        
        return final_audio


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
