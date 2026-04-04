import numpy as np
from scipy import signal
from typing import List

class Crossover:
    """
    Splits an audio signal into multiple bands.
    Uses stateful Linkwitz-Riley (LR8) IIR filters for real-time chunk processing.
    LR8 sums perfectly flat without phase cancellation issues.
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
            # We cascade two 4th order Butterworth filters to create an 8th order Linkwitz-Riley.
            # LR8 ensures flat magnitude sum and in-phase alignment.
            sos_lp = signal.butter(4, cutoff, 'lowpass', fs=self.fs, output='sos')
            sos_hp = signal.butter(4, cutoff, 'highpass', fs=self.fs, output='sos')
            
            self.filters.append({
                'sos_lp': sos_lp,
                'sos_hp': sos_hp,
                'zi_lp1': None,
                'zi_lp2': None,
                'zi_hp1': None,
                'zi_hp2': None
            })

    def update_crossovers(self, crossovers: List[float]):
        """Updates the crossover frequencies and resets state."""
        self.crossovers = sorted(crossovers)
        self.num_bands = len(self.crossovers) + 1
        self._reset_state()

    def split_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Splits a chunk of audio into N bands using stateful filters.
        """
        bands = []
        remainder = chunk

        for f in self.filters:
            # Lowpass path (Band N)
            if f['zi_lp1'] is None:
                n_sections_lp = f['sos_lp'].shape[0]
                if chunk.ndim > 1:
                    f['zi_lp1'] = np.zeros((n_sections_lp, chunk.shape[0], 2))
                    f['zi_lp2'] = np.zeros((n_sections_lp, chunk.shape[0], 2))
                else:
                    f['zi_lp1'] = np.zeros((n_sections_lp, 2))
                    f['zi_lp2'] = np.zeros((n_sections_lp, 2))

            lp_pass1, f['zi_lp1'] = signal.sosfilt(f['sos_lp'], remainder, zi=f['zi_lp1'], axis=-1)
            lp_signal, f['zi_lp2'] = signal.sosfilt(f['sos_lp'], lp_pass1, zi=f['zi_lp2'], axis=-1)

            # Highpass path (Remainder for next bands)
            if f['zi_hp1'] is None:
                n_sections_hp = f['sos_hp'].shape[0]
                if chunk.ndim > 1:
                    f['zi_hp1'] = np.zeros((n_sections_hp, chunk.shape[0], 2))
                    f['zi_hp2'] = np.zeros((n_sections_hp, chunk.shape[0], 2))
                else:
                    f['zi_hp1'] = np.zeros((n_sections_hp, 2))
                    f['zi_hp2'] = np.zeros((n_sections_hp, 2))

            hp_pass1, f['zi_hp1'] = signal.sosfilt(f['sos_hp'], remainder, zi=f['zi_hp1'], axis=-1)
            hp_signal, f['zi_hp2'] = signal.sosfilt(f['sos_hp'], hp_pass1, zi=f['zi_hp2'], axis=-1)

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
