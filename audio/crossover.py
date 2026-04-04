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
        self.filters = []
        self._reset_state()

    def _reset_state(self):
        """Initializes/Resets the filter states for each band split."""
        # Keep existing zi buffers if possible to reduce allocations
        old_filters = self.filters
        self.filters = []
        for i, cutoff in enumerate(self.crossovers):
            # We cascade two 4th order Butterworth filters to create an 8th order Linkwitz-Riley.
            sos_lp = signal.butter(4, cutoff, 'lowpass', fs=self.fs, output='sos')
            sos_hp = signal.butter(4, cutoff, 'highpass', fs=self.fs, output='sos')
            
            f_state = {
                'sos_lp': sos_lp,
                'sos_hp': sos_hp,
                'zi_lp1': None,
                'zi_lp2': None,
                'zi_hp1': None,
                'zi_hp2': None
            }
            
            # Try to reuse zi from old filters if the topology (number of filters) matches
            if i < len(old_filters):
                f_state['zi_lp1'] = old_filters[i]['zi_lp1']
                f_state['zi_lp2'] = old_filters[i]['zi_lp2']
                f_state['zi_hp1'] = old_filters[i]['zi_hp1']
                f_state['zi_hp2'] = old_filters[i]['zi_hp2']
                
                # Zero out existing state instead of re-allocating
                if f_state['zi_lp1'] is not None: f_state['zi_lp1'].fill(0)
                if f_state['zi_lp2'] is not None: f_state['zi_lp2'].fill(0)
                if f_state['zi_hp1'] is not None: f_state['zi_hp1'].fill(0)
                if f_state['zi_hp2'] is not None: f_state['zi_hp2'].fill(0)
                
            self.filters.append(f_state)

    def update_crossovers(self, crossovers: List[float]):
        """Updates the crossover frequencies and resets state."""
        new_crossovers = sorted(crossovers)
        # Only reset if crossovers actually changed or number of bands changed
        if new_crossovers != self.crossovers:
            self.crossovers = new_crossovers
            self.num_bands = len(self.crossovers) + 1
            self._reset_state()

    def split_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Splits a chunk of audio into N bands using stateful filters.
        """
        bands = []
        remainder = chunk
        num_channels = chunk.shape[0] if chunk.ndim > 1 else 1

        for f in self.filters:
            # Lowpass path (Band N)
            n_sections_lp = f['sos_lp'].shape[0]
            expected_shape = (n_sections_lp, num_channels, 2) if chunk.ndim > 1 else (n_sections_lp, 2)
            
            if f['zi_lp1'] is None or f['zi_lp1'].shape != expected_shape:
                f['zi_lp1'] = np.zeros(expected_shape, dtype=np.float32)
                f['zi_lp2'] = np.zeros(expected_shape, dtype=np.float32)

            lp_pass1, f['zi_lp1'] = signal.sosfilt(f['sos_lp'], remainder, zi=f['zi_lp1'], axis=-1)
            lp_signal, f['zi_lp2'] = signal.sosfilt(f['sos_lp'], lp_pass1, zi=f['zi_lp2'], axis=-1)

            # Highpass path (Remainder for next bands)
            n_sections_hp = f['sos_hp'].shape[0]
            if f['zi_hp1'] is None or f['zi_hp1'].shape != expected_shape:
                f['zi_hp1'] = np.zeros(expected_shape, dtype=np.float32)
                f['zi_hp2'] = np.zeros(expected_shape, dtype=np.float32)

            hp_pass1, f['zi_hp1'] = signal.sosfilt(f['sos_hp'], remainder, zi=f['zi_hp1'], axis=-1)
            hp_signal, f['zi_hp2'] = signal.sosfilt(f['sos_hp'], hp_pass1, zi=f['zi_hp2'], axis=-1)

            bands.append(lp_signal)
            remainder = hp_signal
            
        bands.append(remainder)
        return bands

    def sum_bands(self, bands: List[np.ndarray]) -> np.ndarray:
        """Sums the bands back together."""
        if not bands:
            return np.array([], dtype=np.float32)
        
        # Use a more efficient in-place summing if possible
        result = bands[0].copy()
        for i in range(1, len(bands)):
            result += bands[i]
        return result

    # Legacy method for full-file processing (can be used for chunks too but split_chunk is preferred)
    def split(self, audio: np.ndarray) -> List[np.ndarray]:
        self._reset_state()
        return self.split_chunk(audio)
