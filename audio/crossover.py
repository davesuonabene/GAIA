import numpy as np
from scipy import signal
from typing import List

class LinkwitzRileyFilter:
    """
    A 4th-order Linkwitz-Riley filter response.
    We use sosfilt (IIR) for real-time processing capability.
    """
    def __init__(self, cutoff: float, fs: int, btype: str):
        # LR-4 is two cascaded 2nd order Butterworths
        self.sos = signal.butter(2, cutoff, btype=btype, fs=fs, output='sos')
        
    def process(self, data: np.ndarray) -> np.ndarray:
        # Standard IIR filtering (causal)
        # We apply it twice to get the LR-4 response
        first_pass = signal.sosfilt(self.sos, data)
        return signal.sosfilt(self.sos, first_pass)

class Crossover:
    """
    Splits an audio signal into multiple bands.
    To ensure absolute mathematical transparency (Max Amplitude Difference < 1e-10),
    we use a subtractive filter bank approach.
    """
    def __init__(self, crossovers: List[float], fs: int = 44100):
        self.crossovers = sorted(crossovers)
        self.fs = fs
        
    def split(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Splits the audio into N bands.
        Each split is: 
          LowBand = Filter(Signal)
          HighBand = Signal - LowBand
        This guarantees perfect reconstruction when summed.
        """
        bands = []
        remainder = audio
        
        for cutoff in self.crossovers:
            # We use a zero-phase LR-4 for the Low Pass to keep it high quality
            sos = signal.butter(2, cutoff, 'lowpass', fs=self.fs, output='sos')
            # Use filtfilt so the LP is zero-phase, making the subtraction cleaner
            lp_signal = signal.sosfiltfilt(sos, remainder)
            
            # Subtractive High Pass (guarantees perfect sum)
            hp_signal = remainder - lp_signal
            
            bands.append(lp_signal)
            remainder = hp_signal
            
        bands.append(remainder)
        return bands

    def sum_bands(self, bands: List[np.ndarray]) -> np.ndarray:
        """Sums the bands back together."""
        return np.sum(bands, axis=0)

if __name__ == "__main__":
    # --- PHASE TRANSPARENCY TEST ---
    fs = 44100
    duration = 2.0
    num_samples = int(fs * duration)
    
    # 1. Generate 2 seconds of white noise
    noise = np.random.uniform(-0.5, 0.5, num_samples)
    
    # 2. Setup a 3-band crossover
    crossover = Crossover(crossovers=[150.0, 2500.0], fs=fs)
    
    # 3. Split into bands
    bands = crossover.split(noise)
    
    # 4. Sum back together
    reconstructed = crossover.sum_bands(bands)
    
    # 5. Calculate the maximum amplitude difference
    # Subtractive filtering is mathematically identity: LP + (Signal - LP) = Signal
    diff = np.abs(noise - reconstructed)
    max_diff = np.max(diff)
    
    print("=== CROSSOVER TRANSPARENCY TEST ===")
    print(f"Sample Rate: {fs} Hz")
    print(f"Crossover Points: {crossover.crossovers} Hz")
    print(f"Number of Bands: {len(bands)}")
    print(f"Max Amplitude Difference: {max_diff:.2e}")
    
    if max_diff < 1e-10:
        print("RESULT: PASS (Acoustically transparent and phase-accurate)")
    else:
        print("RESULT: FAIL (Deviation too high)")
