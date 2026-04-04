import numpy as np
import pedalboard
import threading
from typing import List, Optional, Dict, Any
import sys
import os
import subprocess
import tempfile
import soundfile as sf
import librosa

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

class HybridChain:
    """A DSP chain that can mix Pedalboard plugins and custom algorithmic modules."""
    def __init__(self, modules, sample_rate: int):
        self.processors = []
        for m in modules:
            plugin = m.get_plugin(sample_rate)
            if plugin:
                self.processors.append(plugin)
            else:
                self.processors.append(m)
                
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        current_audio = audio
        for p in self.processors:
            # Both pedalboard plugins and our AudioModules share the .process(audio, sr) signature
            current_audio = p.process(current_audio, sample_rate)
        return current_audio

class Engine:
    """The core audio engine that maps DNA to DSP in real-time."""
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.lock = threading.Lock()
        
        # Cached DSP components
        self._stem_boards: Dict[str, Optional[HybridChain]] = {}
        self._pre_board: Optional[HybridChain] = None
        self._band_boards: List[Optional[HybridChain]] = []
        self._post_board: Optional[HybridChain] = None
        self._crossover: Optional[Crossover] = None
        
        # State for solo/mute/gain
        self._stem_states: Dict[str, dict] = {}
        self._pre_state = {'soloed': False, 'muted': False, 'bypassed': False, 'gain': 1.0}
        self._band_states = [] # List of dicts: {'solo': bool, 'mute': bool, 'bypassed': bool, 'gain': float}
        self._post_state = {'soloed': False, 'muted': False, 'bypassed': False, 'gain': 1.0}

    def _compile_board(self, band_dna: Band) -> HybridChain:
        """Compiles an AudioModule chain into a HybridChain object."""
        return HybridChain(band_dna.modules, self.sample_rate)

    def update_mix(self, mix_dna: Mix):
        """Compiles the DSP state from the Mix DNA. Thread-safe."""
        with self.lock:
            # 0. Compile Stem bands
            self._stem_boards = {}
            self._stem_states = {}
            for name, band in mix_dna.stem_bands.items():
                self._stem_boards[name] = self._compile_board(band)
                self._stem_states[name] = {
                    'soloed': band.is_soloed,
                    'muted': band.is_muted,
                    'bypassed': getattr(band, 'is_bypassed', False),
                    'gain': 10 ** (band.gain.current_value / 20.0)
                }

            # 1. Compile PRE band
            self._pre_board = self._compile_board(mix_dna.pre_band)
            self._pre_state = {
                'soloed': mix_dna.pre_band.is_soloed,
                'muted': mix_dna.pre_band.is_muted,
                'bypassed': getattr(mix_dna.pre_band, 'is_bypassed', False),
                'gain': 10 ** (mix_dna.pre_band.gain.current_value / 20.0)
            }

            # 2. Compile Parallel bands
            self._band_boards = []
            self._band_states = []
            for band in mix_dna.bands:
                self._band_boards.append(self._compile_board(band))
                self._band_states.append({
                    'soloed': band.is_soloed,
                    'muted': band.is_muted,
                    'bypassed': getattr(band, 'is_bypassed', False),
                    'gain': 10 ** (band.gain.current_value / 20.0)
                })

            # 3. Compile POST band
            self._post_board = self._compile_board(mix_dna.post_band)
            self._post_state = {
                'soloed': mix_dna.post_band.is_soloed,
                'muted': mix_dna.post_band.is_muted,
                'bypassed': getattr(mix_dna.post_band, 'is_bypassed', False),
                'gain': 10 ** (mix_dna.post_band.gain.current_value / 20.0)
            }

            # 4. Update Crossover
            crossover_freqs = [p.current_value for p in mix_dna.crossover_params]
            if self._crossover is None:
                self._crossover = Crossover(crossover_freqs, self.sample_rate)
            else:
                self._crossover.update_crossovers(crossover_freqs)

    def process_chunk(self, input_chunk: Any) -> np.ndarray:
        """
        Processes a small chunk of audio through the compiled DSP chains.
        Input can be an ndarray (standard) or a dict of stem ndarrays.
        """
        with self.lock:
            # 0. Stem Processing (if input is dict)
            if isinstance(input_chunk, dict):
                if not input_chunk:
                    return np.array([])
                
                # Determine target shape from first stem
                first_stem = next(iter(input_chunk.values()))
                num_channels, num_samples = first_stem.shape
                summed = np.zeros((num_channels, num_samples), dtype=np.float32)
                
                solo_active = any(self._stem_states[name]['soloed'] for name in input_chunk)
                
                for name, audio in input_chunk.items():
                    state = self._stem_states.get(name)
                    board = self._stem_boards.get(name)
                    if state is None or board is None: continue
                    
                    if solo_active:
                        if not state['soloed']: continue
                    elif state['muted']:
                        continue
                        
                    # Match length if necessary (pad or truncate)
                    chunk = audio
                    if chunk.shape[1] != num_samples:
                        if chunk.shape[1] > num_samples:
                            chunk = chunk[:, :num_samples]
                        else:
                            temp = np.zeros((num_channels, num_samples), dtype=np.float32)
                            temp[:, :chunk.shape[1]] = chunk
                            chunk = temp
                    
                    # Process and sum
                    # Empty pedalboard acts as a passthrough
                    processed = chunk if state.get('bypassed', False) else board.process(chunk, self.sample_rate)
                    summed += processed * state['gain']
                current_audio = summed
            else:
                current_audio = input_chunk

            if self._pre_board is None:
                return current_audio

            # PRE Solo/Mute logic (Solo doesn't conceptually apply to master buses, but mute does)
            if self._pre_state['muted']:
                current_audio = np.zeros_like(current_audio)
            else:
                # 1. PRE-FX and Gain
                current_audio = current_audio if self._pre_state.get('bypassed', False) else self._pre_board.process(current_audio, self.sample_rate)
                current_audio = current_audio * self._pre_state['gain']

            # 2. Split into bands
            bands_audio = self._crossover.split_chunk(current_audio)
            
            # 3. Check for Solo state
            solo_active = any(state['soloed'] for state in self._band_states)
            
            processed_bands = []
            for i, band_audio in enumerate(bands_audio):
                state = self._band_states[i]
                
                # Solo/Mute logic
                if solo_active:
                    if not state['soloed']:
                        processed_bands.append(np.zeros_like(band_audio))
                        continue
                elif state['muted']:
                    processed_bands.append(np.zeros_like(band_audio))
                    continue

                # Band-FX
                processed_band = band_audio if state.get('bypassed', False) else self._band_boards[i].process(band_audio, self.sample_rate)
                
                # Band-Gain
                processed_bands.append(processed_band * state['gain'])
            
            # 4. Sum bands
            summed_audio = self._crossover.sum_bands(processed_bands)

            if self._post_state['muted']:
                final_audio = np.zeros_like(summed_audio)
            else:
                # 5. POST-FX and Gain
                final_audio = summed_audio if self._post_state.get('bypassed', False) else self._post_board.process(summed_audio, self.sample_rate)
                final_audio = final_audio * self._post_state['gain']

            return final_audio

    def process(self, input_audio: Any, mix_dna: Mix) -> np.ndarray:
        """Processes audio (ndarray or dict of stems) through the Mix DNA."""
        self.update_mix(mix_dna)
        return self.process_chunk(input_audio)

    def separate_stems(self, audio: np.ndarray, sample_rate: int, progress_callback=None, cache_key=None) -> Dict[str, np.ndarray]:
        """
        Separates the input audio into stems using Demucs (htdemucs model).
        Input audio should be in (channels, samples) format.
        Returns a dictionary with stems: vocals, drums, bass, other.
        If cache_key is provided, uses the STEM-CACHE directory to avoid re-separating.
        """
        stem_names = ["vocals", "drums", "bass", "other"]
        cache_dir = None
        output_folder = None
        
        # Determine the base filename for Demucs output
        input_base_name = "temp_input"
        if cache_key:
            input_base_name = os.path.splitext(cache_key)[0]
            cache_dir = os.path.join("STEM-CACHE", cache_key)
            output_folder = os.path.join(cache_dir, "htdemucs", input_base_name)
            
            # Check if all stems exist in cache
            if os.path.exists(output_folder):
                all_exist = all(os.path.exists(os.path.join(output_folder, f"{stem}.wav")) for stem in stem_names)
                if all_exist:
                    # Load from cache
                    stems = {}
                    for stem in stem_names:
                        stem_path = os.path.join(output_folder, f"{stem}.wav")
                        stem_arr, stem_sr = sf.read(stem_path)
                        stem_arr = stem_arr.T
                        
                        # Resample if mismatch
                        if int(stem_sr) != int(sample_rate):
                            stem_arr = librosa.resample(stem_arr, orig_sr=stem_sr, target_sr=sample_rate)
                        
                        # Ensure exact length match with input audio
                        target_len = audio.shape[1]
                        if stem_arr.shape[1] > target_len:
                            stem_arr = stem_arr[:, :target_len]
                        elif stem_arr.shape[1] < target_len:
                            pad = np.zeros((stem_arr.shape[0], target_len - stem_arr.shape[1]), dtype=stem_arr.dtype)
                            stem_arr = np.concatenate([stem_arr, pad], axis=1)
                            
                        stems[stem] = np.ascontiguousarray(stem_arr)
                    
                    # Instantly set progress to 100% since it's cached
                    if progress_callback:
                        progress_callback(100.0)
                    return stems
                    
            os.makedirs(cache_dir, exist_ok=True)
            temp_dir = cache_dir
        else:
            # If no cache key, create a temporary directory object which we must keep alive
            _temp_obj = tempfile.TemporaryDirectory()
            temp_dir = _temp_obj.name

        try:
            temp_input_path = os.path.join(temp_dir, f"{input_base_name}.wav")
            
            # Write input file (soundfile expects (samples, channels))
            try:
                sf.write(temp_input_path, audio.T, samplerate=sample_rate)
            except Exception as e:
                raise Exception(f"Failed to write temporary input file: {str(e)}")

            # Run Demucs with streaming output to catch progress
            try:
                process = subprocess.Popen(
                    ["demucs", "-n", "htdemucs", temp_input_path, "-o", temp_dir],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, # tqdm sometimes goes to stdout or stderr depending on pipe
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Demucs prints its tqdm progress bar using carriage returns (\r)
                # We need to read char by char to catch these live updates instead of waiting for a \n
                buffer = ""
                while True:
                    char = process.stdout.read(1)
                    if not char and process.poll() is not None:
                        break
                    
                    if char in ('\r', '\n'):
                        if progress_callback and "%|" in buffer:
                            try:
                                parts = buffer.split("%|")
                                if len(parts) > 1:
                                    perc_str = parts[0].split()[-1].replace('%', '').strip()
                                    progress_callback(float(perc_str))
                            except Exception:
                                pass
                        buffer = ""
                    else:
                        buffer += char
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
                    
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                error_msg = getattr(e, 'stderr', str(e))
                raise Exception(f"Demucs failed or is not installed: {error_msg}")

            # Read results
            stems = {}
            if output_folder is None:
                output_folder = os.path.join(temp_dir, "htdemucs", input_base_name)
                
            for stem in stem_names:
                stem_path = os.path.join(output_folder, f"{stem}.wav")
                if not os.path.exists(stem_path):
                    raise Exception(f"Demucs output missing for {stem}: {stem_path}")
                
                # sf.read returns (samples, channels), we need (channels, samples)
                stem_arr, stem_sr = sf.read(stem_path)
                stem_arr = stem_arr.T
                
                # Resample if mismatch
                if int(stem_sr) != int(sample_rate):
                    stem_arr = librosa.resample(stem_arr, orig_sr=stem_sr, target_sr=sample_rate)
                    
                # Ensure exact length match with input audio
                target_len = audio.shape[1]
                if stem_arr.shape[1] > target_len:
                    stem_arr = stem_arr[:, :target_len]
                elif stem_arr.shape[1] < target_len:
                    pad = np.zeros((stem_arr.shape[0], target_len - stem_arr.shape[1]), dtype=stem_arr.dtype)
                    stem_arr = np.concatenate([stem_arr, pad], axis=1)

                stems[stem] = np.ascontiguousarray(stem_arr)

            return stems
        finally:
            if not cache_key:
                # Cleanup if using a temporary directory
                try:
                    _temp_obj.cleanup()
                except Exception:
                    pass
