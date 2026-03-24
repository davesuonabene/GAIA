import numpy as np
from scipy import signal
from typing import List

class Crossover:
    """
    Splits an audio signal into multiple bands.
    Uses stateful IIR filters for real-time chunk processing.
    """
    def __init__(self, crossovers: List[float], fs: int = 44100):
        self.crossovers = sorted(crossovers)
        self.fs = fs
        self.num_bands = len(self.crossovers) + 1
        self._reset_state()

    def _reset_state(self):
        """Initializes/Resets the filter states for each band split."""
        self.filters = []
        for cutoff in self.crossovers:
            # We use a 4th order Linkwitz-Riley (cascaded 2nd order Butterworth)
            # LR-4 has a flat summed frequency response.
            sos = signal.butter(2, cutoff, 'lowpass', fs=self.fs, output='sos')
            # Each split needs its own state (zi)
            # Since it's LR-4 (two passes), we need two states if we were doing it manually,
            # but we can just use a 4th order Butterworth SOS if we want LR-4 behavior? 
            # Actually LR-4 is (Butterworth-2)^2.
            
            # For simplicity and perfect reconstruction in real-time without complex phase alignment:
            # We'll use the subtractive approach but with stateful sosfilt.
            self.filters.append({
                'sos': sos,
                'zi1': None, # State for first pass
                'zi2': None  # State for second pass
            })

    def update_crossovers(self, crossovers: List[float]):
        """Updates the crossover frequencies and resets state."""
        self.crossovers = sorted(crossovers)
        self.num_bands = len(self.crossovers) + 1
        self._reset_state()

    def split_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Splits a chunk of audio into N bands using stateful filters.
        Guarantees perfect reconstruction when summed due to subtractive architecture.
        """
        bands = []
        remainder = chunk

        for f in self.filters:
            # First pass
            if f['zi1'] is None:
                # Initialize state with zeros. shape of zi is (n_sections, channels, 2)
                f['zi1'] = np.zeros((f['sos'].shape[0], 2))
                if chunk.ndim > 1: # Stereo or Multi-channel
                    f['zi1'] = np.zeros((f['sos'].shape[0], chunk.shape[0], 2))

            lp_pass1, f['zi1'] = signal.sosfilt(f['sos'], remainder, zi=f['zi1'], axis=-1)

            # Second pass (to make it 4th order / LR-4 equivalent)
            if f['zi2'] is None:
                f['zi2'] = np.zeros((f['sos'].shape[0], 2))
                if chunk.ndim > 1:
                    f['zi2'] = np.zeros((f['sos'].shape[0], chunk.shape[0], 2))

            lp_signal, f['zi2'] = signal.sosfilt(f['sos'], lp_pass1, zi=f['zi2'], axis=-1)            
            # Subtractive High Pass
            hp_signal = remainder - lp_signal
            
            bands.append(lp_signal)
            remainder = hp_signal
            
        bands.append(remainder)
        return bands

    def sum_bands(self, bands: List[np.ndarray]) -> np.ndarray:
        """Sums the bands back together."""
        return np.sum(bands, axis=0)

    # Legacy method for full-file processing (can be used for chunks too but split_chunk is preferred)
    def split(self, audio: np.ndarray) -> List[np.ndarray]:
        self._reset_state()
        return self.split_chunk(audio)
