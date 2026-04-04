import sys
import os
import argparse
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
import threading
from typing import List, Any, Callable, Optional, Dict
from enum import Enum, auto
from dataclasses import dataclass, field

# Ensure we can import from core and ga
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.metadata import TrackMetadata
from core.preset_manager import PresetManager

class FocusZone(Enum):
    MENU = auto()
    POPULATION = auto()
    BANDS = auto()
    PLAYBACK = auto()
    STEMS = auto()

@dataclass
class MenuItem:
    title: str
    action: Optional[Callable] = None
    children: List['MenuItem'] = field(default_factory=list)

class HeaderMenu:
    def __init__(self, items: List[MenuItem] = None):
        self.items = items or []
        self.selected_index = 0


# Linux-specific imports for terminal management
try:
    import tty
    import termios
    import select
except ImportError:
    pass

from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich import box
from rich.layout import Layout
from rich.table import Table
from rich.progress import ProgressBar

from core.mix import Mix
from core.band import Band
from core.parameter import Parameter
from ga.population import Population
from core.audio_module import (
    CompressorModule, ExpanderModule, TransientShaperModule, 
    SaturationModule, ClipperModule, LimiterModule, ConvolutionModule
)
from audio.engine import Engine

# Constants
KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_RIGHT = "RIGHT"
KEY_LEFT = "LEFT"
KEY_SHIFT_UP = "SHIFT_UP"
KEY_SHIFT_DOWN = "SHIFT_DOWN"
KEY_CTRL_UP = "CTRL_UP"
KEY_CTRL_DOWN = "CTRL_DOWN"
KEY_CTRL_RIGHT = "CTRL_RIGHT"
KEY_CTRL_LEFT = "CTRL_LEFT"
KEY_ENTER = "ENTER"
KEY_TAB = "TAB"
KEY_CTRL_C = "CTRL_C"
KEY_ESC = "ESC"
KEY_SPACE = " "
KEY_BACKSPACE = "BACKSPACE"
KEY_CTRL_BACKSPACE = "CTRL_BACKSPACE"
KEY_L = "L"
KEY_E = "E"
KEY_P = "P"
KEY_M = "M"
KEY_S = "S"
KEY_K = "K"
KEY_D = "D"
KEY_B = "B"
KEY_BRACKET_LEFT = "["
KEY_BRACKET_RIGHT = "]"

class TerminalMode:
    """Context manager to handle Linux raw terminal mode without flickering."""
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = None

    def __enter__(self):
        try:
            self.old_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except Exception:
            pass
        return self

    def __exit__(self, type, value, traceback):
        if self.old_settings:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

def get_key(fd):
    """Robust non-blocking key reader for Linux escape sequences."""
    r, _, _ = select.select([fd], [], [], 0.02)
    if not r:
        return None
    
    try:
        b = os.read(fd, 8)
    except EOFError:
        return None

    # Escape sequences
    if b == b'\x1b[A': return KEY_UP
    if b == b'\x1b[B': return KEY_DOWN
    if b == b'\x1b[C': return KEY_RIGHT
    if b == b'\x1b[D': return KEY_LEFT
    
    # SHIFT sequences
    if b == b'\x1b[1;2A': return KEY_SHIFT_UP
    if b == b'\x1b[1;2B': return KEY_SHIFT_DOWN

    # CTRL sequences
    if b == b'\x1b[1;5A': return KEY_CTRL_UP
    if b == b'\x1b[1;5B': return KEY_CTRL_DOWN
    if b == b'\x1b[1;5C': return KEY_CTRL_RIGHT
    if b == b'\x1b[1;5D': return KEY_CTRL_LEFT
    
    if b == b'\x1b': return KEY_ESC
    if b in (b'\r', b'\n'): return KEY_ENTER
    if b == b'\t': return KEY_TAB
    if b == b'\x03': return KEY_CTRL_C
    if b == b'\x7f': return KEY_BACKSPACE
    if b in (b'\x08', b'\x17'): return KEY_CTRL_BACKSPACE
    if b == b' ': return KEY_SPACE
    
    try:
        decoded = b.decode('ascii').upper()
        if decoded == '[': return KEY_BRACKET_LEFT
        if decoded == ']': return KEY_BRACKET_RIGHT
        if decoded == 'M': return KEY_M
        if decoded == 'S': return KEY_S
        if decoded == 'K': return KEY_K
        if decoded == 'D': return KEY_D
        return decoded
    except Exception:
        return None

class GaiaTUI:
    def __init__(self, population: Population, audio_data: np.ndarray, sample_rate: int, output_dir: str, blocksize: int = 512):
        self.population = population
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.blocksize = blocksize
        
        # Navigation State
        self.focus_zone = FocusZone.POPULATION
        self.header_menu = HeaderMenu(items=[
            MenuItem(title="File", children=[
                MenuItem(title="Swap Input Track", action=self.action_swap_track),
                MenuItem(title="Load Preset", action=self.action_load_preset_flow),
                MenuItem(title="Save Preset", action=self.action_save_preset_flow),
                MenuItem(title="Export Track", action=self.action_export_track_flow),
            ]),
            MenuItem(title="Stems", children=[
                MenuItem(title="Separate Stems", action=self.action_separate_stems),
            ]),
            MenuItem(title="Settings", children=[
                MenuItem(title="Audio Blocksize", action=self.action_change_blocksize),
            ]),
        ])
        self.metadata = TrackMetadata()
        self.current_filename = None
        self.selected_mix_idx = 0
        self.selected_file_idx = 0 # Separate index for file browser
        self.selected_band_idx = 0
        self.selected_module_idx = 0
        self.selected_param_idx = 0
        self.editing_crossover = False
        
        # Stem Navigation Cursors
        self.stem_col_idx = 0
        self.stem_mod_idx = 0
        self.stem_param_idx = 0
        
        # Stem Separation State
        self.is_separating = False
        self.separation_progress = 0.0
        self.stems_data = None
        self.selected_stem_idx = 0 # DEPRECATED, will be replaced by stem_col_idx but keeping for safety if used elsewhere
        self.stem_names = ["vocals", "drums", "bass", "other"]
        
        # FX Selection Mode
        self.selecting_new_fx = False
        self.available_fx_pool = []
        
        # Dynamic Input Mode
        self.editing = False
        self.edit_buffer = ""
        self.input_mode = None # "SAVE_PRESET", "EXPORT_TRACK", or None
        
        # Export Modal State
        self.in_export_modal = False
        self.export_opt_full = True
        self.export_opt_stems = False
        self.export_modal_idx = 0

        # Sub-menu State
        self.in_submenu = False
        self.selected_child_idx = 0
        
        self.confirming_delete = False
        
        self.running = True
        self.engine = Engine(sample_rate=self.sample_rate) if self.sample_rate else None
        self.is_playing = False
        self.playback_duration = 0
        
        # Audio Streaming State
        self.stream = None
        self.preview_stream = None
        self.play_idx = 0
        
        # Buffers for real-time safety in audio_callback
        self._raw_chunk_buffer: Optional[np.ndarray] = None
        self._stem_chunk_buffers: Dict[str, np.ndarray] = {}
        
        # Mode and File Browser
        self.mode = "EVOLUTION" if self.audio_data is not None else "FILE_PICKER"
        self.current_path = os.path.abspath(".")
        self.file_list = []
        if self.mode == "FILE_PICKER":
            self.refresh_file_list()
        
        # Inline Editing State
        self.editing = False
        self.edit_buffer = ""
        self.status_msg = "Arrows: Navigate/Adjust | TAB: Zone | CTRL+Arrows: Granular/Jump | SPACE: Play/Stop"

    def refresh_file_list(self):
        """Scans path for audio and preset files."""
        exts = (".wav", ".flac", ".mp3", ".aiff", ".ogg", ".json")
        try:
            items = os.listdir(self.current_path)
            self.file_list = [{"name": "..", "type": "dir"}] if os.path.dirname(self.current_path) != self.current_path else []
            self.file_list += [{"name": d, "type": "dir"} for d in sorted(items) if os.path.isdir(os.path.join(self.current_path, d)) and not d.startswith(".")]
            self.file_list += [{"name": f, "type": "file"} for f in sorted(items) if f.lower().endswith(exts)]
            self.selected_file_idx = 0
        except Exception as e:
            self.status_msg = f"Error: {e}"

    def action_export_track_flow(self):
        """Initiate the export flow."""
        if self.audio_data is None:
            self.status_msg = "Nothing to export."
            return
        self.in_export_modal = True
        self.export_modal_idx = 0
        if self.stems_data is None:
            self.export_opt_stems = False
        self.status_msg = "Configure export settings."

    def action_swap_track(self):
        """Reset state and go back to File Picker."""
        if self.is_playing:
            self.toggle_playback()
        self.audio_data = None
        self.engine = None
        self.mode = "FILE_PICKER"
        self.status_msg = "Pick a new audio track."
        self.refresh_file_list()

    def action_change_blocksize(self):
        """Cycle through common audio blocksizes."""
        sizes = [256, 512, 1024, 2048, 4096]
        try:
            idx = sizes.index(self.blocksize)
            self.blocksize = sizes[(idx + 1) % len(sizes)]
        except ValueError:
            self.blocksize = 512
        
        self.status_msg = f"Blocksize changed to: {self.blocksize}"
        if self.is_playing:
            self.toggle_playback()
            self.toggle_playback()

    def action_save_preset_flow(self):
        """Initiate the save preset flow."""
        self.editing = True
        self.input_mode = "SAVE_PRESET"
        self.edit_buffer = ""
        self.status_msg = "Enter preset name: █ (ENTER to save, ESC to cancel)"

    def action_load_preset_flow(self):
        """Open the file picker in the presets directory."""
        os.makedirs("./presets", exist_ok=True)
        self.mode = "FILE_PICKER"
        self.current_path = os.path.abspath("./presets")
        self.refresh_file_list()
        self.status_msg = "Arrows: Select Preset | ENTER: Load | ESC: Cancel"

    def action_separate_stems(self):
        if self.audio_data is None:
            self.status_msg = "No audio loaded to separate."
            return
        if self.is_separating:
            self.status_msg = "Already separating stems..."
            return
            
        self.is_separating = True
        self.status_msg = "Separating Stems... (This takes 1-3 minutes)"
        threading.Thread(target=self._separation_worker, daemon=True).start()

    def _separation_worker(self):
        def _update_progress(percent: float):
            self.separation_progress = percent
            
        try:
            self.separation_progress = 0.0
            cache_key = self.current_filename if self.current_filename else None
            stems = self.engine.separate_stems(self.audio_data, self.sample_rate, progress_callback=_update_progress, cache_key=cache_key)
            self.stems_data = stems
            self.status_msg = f"Separation Complete! Stems cached in STEM-CACHE/{cache_key}" if cache_key else "Separation Complete!"
            # Trigger mix update to refresh audio buffer
            if self.is_playing:
                idx = self.selected_mix_idx % len(self.population.mixes)
                self.engine.update_mix(self.population.mixes[idx])
        except Exception as e:
            self.status_msg = f"Separation Failed: {str(e)}"
        finally:
            self.is_separating = False
            self.separation_progress = 0.0

    def _preview_selected_file(self):
        """Starts a background playback of the selected file for preview."""
        if self.preview_stream:
            try:
                self.preview_stream.stop()
            except Exception:
                pass
            self.preview_stream = None
            sd.stop()

        if self.selected_file_idx >= len(self.file_list):
            return

        item = self.file_list[self.selected_file_idx]
        if item["type"] == "dir":
            return

        exts = (".wav", ".flac", ".mp3", ".aiff", ".ogg")
        if not any(item["name"].lower().endswith(e) for e in exts):
            return

        class PreviewHandle:
            def stop(self):
                sd.stop()

        def _play_thread():
            try:
                path = os.path.join(self.current_path, item["name"])
                data, sr = sf.read(path)
                # Keep a short preview, say 5 seconds
                preview_len = int(sr * 5)
                # Ensure data is (samples, channels) for sd.play
                if data.ndim == 1:
                    data = data[:preview_len]
                else:
                    data = data[:preview_len, :]
                
                sd.play(data, sr)
                self.preview_stream = PreviewHandle()
            except Exception:
                pass

        threading.Thread(target=_play_thread, daemon=True).start()

    def audio_callback(self, outdata, frames, time_info, status):
        if self.audio_data is None:
            outdata.fill(0)
            return

        # 1. Determine Source length
        source_len = self.audio_data.shape[1]
        if self.stems_data is not None:
            source_len = min([s.shape[1] for s in self.stems_data.values()])

        # Helper to fetch and slice a chunk (Modified for pre-allocation)
        def fetch_chunk(start_idx, count):
            if self.stems_data is None:
                return self.audio_data[:, start_idx:start_idx+count]
            else:
                # We can't easily return a view of multiple stems in a dict without allocation,
                # but we can at least avoid re-creating the dict if we were careful.
                # For now, let's just ensure the underlying arrays are views.
                return {name: stem_audio[:, start_idx:start_idx+count] 
                        for name, stem_audio in self.stems_data.items()}

        # 2. Slice current chunk with loop handling
        try:
            if self.play_idx + frames <= source_len:
                raw_chunk = fetch_chunk(self.play_idx, frames)
                self.play_idx += frames
            else:
                # Wrap around logic
                n1 = max(0, source_len - self.play_idx)
                n2 = frames - n1

                if self.stems_data is None:
                    if self._raw_chunk_buffer is None or self._raw_chunk_buffer.shape != (self.audio_data.shape[0], frames):
                        self._raw_chunk_buffer = np.zeros((self.audio_data.shape[0], frames), dtype=np.float32)

                    self._raw_chunk_buffer.fill(0)
                    if n1 > 0:
                        self._raw_chunk_buffer[:, :n1] = fetch_chunk(self.play_idx, n1)

                    self.play_idx = 0
                    if n2 > 0:
                        fill_size = min(n2, source_len)
                        self._raw_chunk_buffer[:, n1:n1+fill_size] = fetch_chunk(0, fill_size)
                        self.play_idx = fill_size
                    raw_chunk = self._raw_chunk_buffer
                else:
                    # Wrap around for stems
                    num_channels = next(iter(self.stems_data.values())).shape[0]
                    raw_chunk = {}
                    for name in self.stems_data:
                        if name not in self._stem_chunk_buffers or self._stem_chunk_buffers[name].shape != (num_channels, frames):
                            self._stem_chunk_buffers[name] = np.zeros((num_channels, frames), dtype=np.float32)

                        stem_raw = self._stem_chunk_buffers[name]
                        stem_raw.fill(0)

                        if n1 > 0:
                            stem_raw[:, :n1] = self.stems_data[name][:, self.play_idx:self.play_idx+n1]

                        if n2 > 0:
                            fill_size = min(n2, source_len)
                            stem_raw[:, n1:n1+fill_size] = self.stems_data[name][:, 0:fill_size]

                        raw_chunk[name] = stem_raw

                    self.play_idx = n2 % source_len if n2 > 0 else 0

            # 3. Process chunk through active DSP state
            # engine.process_chunk handles the dict or ndarray
            processed_chunk = self.engine.process_chunk(raw_chunk)

            # 4. Handle channel mismatch (e.g., stereo stems on mono output)
            if processed_chunk.shape[0] != outdata.shape[1]:
                if outdata.shape[1] == 1:
                    processed_chunk = np.mean(processed_chunk, axis=0, keepdims=True)
                elif processed_chunk.shape[0] == 1:
                    processed_chunk = np.tile(processed_chunk, (outdata.shape[1], 1))
                else:
                    # Truncate or pad if they are both multi-channel but different
                    mismatch_buf = self.engine._get_buffer("_mismatch_buf", (outdata.shape[1], processed_chunk.shape[1]))
                    mismatch_buf.fill(0.0)
                    min_ch = min(outdata.shape[1], processed_chunk.shape[0])
                    mismatch_buf[:min_ch, :] = processed_chunk[:min_ch, :]
                    processed_chunk = mismatch_buf

            # 4.5 Apply invisible -1dB (-10.8% amplitude) output scaling to give stream headroom and prevent pipeline clipping
            headroom_multiplier = 10 ** (-1.0 / 20.0)
            processed_chunk = processed_chunk * headroom_multiplier

            # 5. Assign to output and hard clip to prevent integer wrap/crackle
            outdata[:] = np.clip(processed_chunk.T, -1.0, 1.0)
        except Exception:
            outdata.fill(0)
    def load_audio(self, filename: str):
        if self.is_playing:
            self.toggle_playback() # stop playback safely
        path = os.path.join(self.current_path, filename)
        try:
            data, sr = sf.read(path)
            self.audio_data = np.ascontiguousarray(data.T if data.ndim > 1 else data.reshape(1, -1))
            self.sample_rate = sr
            self.current_filename = os.path.basename(filename)
            self.engine = Engine(sample_rate=self.sample_rate)
            
            # Auto-load stems if they exist in cache
            cache_dir = os.path.join("STEM-CACHE", self.current_filename, "htdemucs", os.path.splitext(self.current_filename)[0])
            if os.path.exists(cache_dir):
                stem_names = ["vocals", "drums", "bass", "other"]
                if all(os.path.exists(os.path.join(cache_dir, f"{s}.wav")) for s in stem_names):
                    self.stems_data = {}
                    for s in stem_names:
                        arr, stem_sr = sf.read(os.path.join(cache_dir, f"{s}.wav"))
                        arr = arr.T

                        if int(stem_sr) != int(self.sample_rate):
                            arr = librosa.resample(arr, orig_sr=stem_sr, target_sr=self.sample_rate)

                        # Ensure exact length match
                        target_len = self.audio_data.shape[1]
                        if arr.shape[1] > target_len:
                            arr = arr[:, :target_len]
                        elif arr.shape[1] < target_len:
                            pad = np.zeros((arr.shape[0], target_len - arr.shape[1]), dtype=arr.dtype)
                            arr = np.concatenate([arr, pad], axis=1)

                        self.stems_data[s] = np.ascontiguousarray(arr)
                    self.status_msg = f"Loaded {filename} (with cached stems resampled to {self.sample_rate}Hz)"

                else:
                    self.stems_data = None
            else:
                self.stems_data = None

            # Initial mix compilation
            if len(self.population.mixes) > 0:
                self.engine.update_mix(self.population.mixes[0])
            
            self.mode = "EVOLUTION"
            self.selected_mix_idx = 0 # Reset to prevent IndexError
            self.play_idx = 0
            
            # Update Track Metadata
            duration_sec = self.audio_data.shape[1] / sr
            self.metadata.filename = filename
            self.metadata.sample_rate = sr
            self.metadata.channels = self.audio_data.shape[0]
            self.metadata.duration_sec = duration_sec
            self.playback_duration = duration_sec
            
            self.status_msg = f"Loaded {filename}"
        except Exception as e:
            self.status_msg = f"Error: {e}"

    def toggle_playback(self):
        if self.audio_data is None: return
        if self.is_playing:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.is_playing = False
            self.status_msg = "Stopped."
        else:
            try:
                # Compile current mix immediately
                idx = self.selected_mix_idx % len(self.population.mixes)
                self.engine.update_mix(self.population.mixes[idx])
                
                # Low-latency settings for Pipewire/Linux
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate, 
                    channels=self.audio_data.shape[0], 
                    callback=self.audio_callback,
                    blocksize=self.blocksize
                )
                self.stream.start()
                self.is_playing = True
                self.status_msg = "Playing (Real-time DSP)..."
            except Exception as e:
                self.status_msg = f"Error: {e}"

    def get_column_data(self, depth: int) -> List[Any]:
        """Returns the data list for a specific depth in the navigation tree."""
        idx = self.selected_mix_idx % len(self.population.mixes)
        mix = self.population.mixes[idx]
        
        if self.focus_zone == FocusZone.STEMS:
            if depth == 1:
                return [mix.stem_bands[name] for name in self.stem_names]
            if depth == 2:
                if self.selecting_new_fx:
                    return self.available_fx_pool
                col_data = self.get_column_data(1)
                if 0 <= self.stem_col_idx < len(col_data):
                    parent = col_data[self.stem_col_idx]
                    return parent.modules + ["[+] Add FX"]
            return []

        if depth == 1:
            # Tree conceptually: [pre_band, ...bands, post_band]
            return [mix.pre_band] + mix.bands + [mix.post_band]
        
        if depth == 2:
            # Check if we are in FX selection mode
            if self.selecting_new_fx:
                return self.available_fx_pool

            # Get the focused item at depth 1
            all_d1 = self.get_column_data(1)
            if self.selected_band_idx < len(all_d1):
                parent = all_d1[self.selected_band_idx]
                if isinstance(parent, Band):
                    # Band FX Chain + Dummy [+] Button
                    return parent.modules + ["[+] Add FX"]
        return []

    def render_column_content(self, column_data: List[Any], depth: int, is_focused: bool = False) -> Group:
        """Renders the content of a column for display in Depth 2 (Modules list)."""
        items = []
        
        if depth == 2:
            if self.selecting_new_fx and is_focused:
                items.append(Text(" SELECT FX TO ADD: ", style="bold white on green"))
                items.append(Text("-" * 20, style="dim"))

            # Determine which cursors to use
            cur_mod_idx = self.stem_mod_idx if self.focus_zone == FocusZone.STEMS else self.selected_module_idx
            cur_param_idx = self.stem_param_idx if self.focus_zone == FocusZone.STEMS else self.selected_param_idx

            for m_idx, item in enumerate(column_data):
                is_mod_sel = is_focused and (not self.editing_crossover) and cur_mod_idx == m_idx
                
                # Handling Class types (for selection mode)
                if isinstance(item, type):
                    name = item().name.upper() # Safe spawn just for name
                    style = "bold white on green" if is_mod_sel else "green"
                    items.append(Text(f" > {name}", style=style))
                    continue

                # Resilience: handle dummy string [+] Add FX
                if isinstance(item, str):
                    style = "bold green" if is_mod_sel else "green"
                    items.append(Text(f" {item}", style=style))
                    continue
                
                # Resilience: handle Crossover parameters
                if isinstance(item, Parameter):
                    # For crossovers, we just show the name and value
                    val_txt = self.edit_buffer if (is_mod_sel and self.editing) else f"{item.current_value:.0f}Hz"
                    if item.is_locked: val_txt = f"*{val_txt}"
                    style = "bold black on yellow" if (is_mod_sel and self.editing) else "bold white on cyan" if is_mod_sel else "dim"
                    items.append(Text(f" {item.name}: {val_txt}", style=style))
                    continue

                # Normal AudioModule rendering
                mod_name = item.name.upper()
                style = "bold white on blue" if is_mod_sel else "white"
                items.append(Text(f"[{mod_name}]", style=style, no_wrap=True))
                
                # Render parameters ONLY if selected (Accordion Logic)
                if is_mod_sel:
                    for p_idx, (p_name, p) in enumerate(item.parameters.items()):
                        is_p_sel = is_mod_sel and cur_param_idx == p_idx
                        val_txt = self.edit_buffer if (is_p_sel and self.editing) else f"{p.current_value:.1f}"
                        if p.is_locked: val_txt = f"*{val_txt}"
                        p_style = "bold black on yellow" if (is_p_sel and self.editing) else "bold white on cyan" if is_p_sel else "dim"
                        items.append(Align.right(Text(f" {p_name}: {val_txt}", style=p_style, overflow="ellipsis", no_wrap=True)))
        
        return Group(*items)

    def get_selected_param(self):
        if self.mode != "EVOLUTION" or self.selecting_new_fx: return None
        idx = self.selected_mix_idx % len(self.population.mixes)
        mix = self.population.mixes[idx]
        
        if self.focus_zone == FocusZone.STEMS:
            cols = self.get_column_data(1)
            if self.stem_col_idx >= len(cols): return None
            parent = cols[self.stem_col_idx]
            
            modules_list = self.get_column_data(2)
            self.stem_mod_idx = max(-1, min(self.stem_mod_idx, len(modules_list) - 1))
            
            if self.stem_mod_idx == -1 and isinstance(parent, Band):
                return parent.gain
            
            if 0 <= self.stem_mod_idx < len(modules_list):
                item = modules_list[self.stem_mod_idx]
                if hasattr(item, "parameters"):
                    params = list(item.parameters.values())
                    self.stem_param_idx = max(0, min(self.stem_param_idx, max(0, len(params) - 1)))
                    return params[self.stem_param_idx] if len(params) > 0 else None
            return None

        # New logic using get_column_data (Depth 1)
        cols = self.get_column_data(1)
        if self.selected_band_idx >= len(cols): return None
        parent = cols[self.selected_band_idx]
        
        # Check if we are on a frequency crossover
        if self.editing_crossover:
            if 1 <= self.selected_band_idx <= len(mix.crossover_params):
                xo_idx = self.selected_band_idx - 1
                return mix.crossover_params[xo_idx]
            else:
                self.editing_crossover = False # Auto-correct if on wrong band
                return None
            
        # Depth 2: Either Module Parameter or Band Gain
        modules_list = self.get_column_data(2)
        
        # Strictly clamp selected_module_idx to prevent IndexError from GA removals
        self.selected_module_idx = max(-1, min(self.selected_module_idx, len(modules_list) - 1))
        
        if self.selected_module_idx == -1 and isinstance(parent, Band):
            return parent.gain
            
        if 0 <= self.selected_module_idx < len(modules_list):
            item = modules_list[self.selected_module_idx]
            
            # If it's an AudioModule (Band column)
            if hasattr(item, "parameters"):
                params = list(item.parameters.values())
                # Strictly clamp selected_param_idx
                self.selected_param_idx = max(0, min(self.selected_param_idx, max(0, len(params) - 1)))
                return params[self.selected_param_idx] if len(params) > 0 else None
        
        return None

    def adjust_value(self, direction: int, granular: bool = False):
        p = self.get_selected_param()
        if not p: return

        idx = self.selected_mix_idx % len(self.population.mixes)
        mix = self.population.mixes[idx]
        
        # Exponential scaling for crossovers: logarithmic frequency navigation
        if p in mix.crossover_params:
            # Factor for 2% change (standard) or 0.2% change (granular)
            # This ensures sub-hertz precision in the lows (e.g., 0.4Hz steps at 20Hz)
            # and natural octave-based navigation in the highs.
            factor = 1.02 if not granular else 1.002
            
            if direction > 0:
                new_value = p.current_value * factor
            else:
                new_value = p.current_value / factor
            
            # Dynamic bounds based on neighbors
            sorted_xo = sorted(mix.crossover_params, key=lambda x: x.current_value)
            p_idx = sorted_xo.index(p)
            
            # Use small buffer to avoid total collapse
            dyn_min = sorted_xo[p_idx - 1].current_value + 0.1 if p_idx > 0 else p.min_bound
            dyn_max = sorted_xo[p_idx + 1].current_value - 0.1 if p_idx < len(sorted_xo) - 1 else p.max_bound
            
            p.current_value = max(dyn_min, min(dyn_max, new_value))
            mix.sort_crossovers()
        else:
            # Normal linear parameter adjustment for other DSP modules
            step = (p.max_bound - p.min_bound) * 0.05 * direction
            if granular:
                step /= 5.0
            p.current_value = max(p.min_bound, min(p.max_bound, p.current_value + step))
            
        # Refresh engine if playing
        if self.is_playing:
            self.engine.update_mix(mix)

    def set_value(self, val: float):
        p = self.get_selected_param()
        if not p: return

        # Get context for crossovers
        idx = self.selected_mix_idx % len(self.population.mixes)
        mix = self.population.mixes[idx]

        if p in mix.crossover_params:
            sorted_xo = sorted(mix.crossover_params, key=lambda x: x.current_value)
            p_idx = sorted_xo.index(p)
            
            # Dynamic bounds based on neighbors
            dyn_min = sorted_xo[p_idx - 1].current_value + 1.0 if p_idx > 0 else p.min_bound
            dyn_max = sorted_xo[p_idx + 1].current_value - 1.0 if p_idx < len(sorted_xo) - 1 else p.max_bound
            
            # Revert if out of dynamic range
            if val < dyn_min or val > dyn_max:
                self.status_msg = f"Value out of band range! Allowed: {dyn_min:.0f}-{dyn_max:.0f} Hz"
                return
            
            p.current_value = val
            mix.sort_crossovers()
        else:
            # Normal parameter set
            p.current_value = max(p.min_bound, min(p.max_bound, val))
            
        if self.is_playing:
            self.engine.update_mix(mix)

    def execute_action(self):
        """Executes the primary action (ENTER) based on the current focus."""
        if self.confirming_delete:
            cols = self.get_column_data(1)
            # Use correct cursors based on zone
            cur_col_idx = self.stem_col_idx if self.focus_zone == FocusZone.STEMS else self.selected_band_idx
            cur_mod_idx = self.stem_mod_idx if self.focus_zone == FocusZone.STEMS else self.selected_module_idx
            
            if cur_col_idx < len(cols):
                band = cols[cur_col_idx]
                if isinstance(band, Band):
                    try:
                        removed = band.modules.pop(cur_mod_idx)
                        self.status_msg = f"Deleted {removed.name}."
                        if self.focus_zone == FocusZone.STEMS:
                            self.stem_mod_idx = max(0, self.stem_mod_idx - 1)
                            self.stem_param_idx = 0
                        else:
                            self.selected_module_idx = max(0, self.selected_module_idx - 1)
                            self.selected_param_idx = 0
                        if self.is_playing:
                            idx = self.selected_mix_idx % len(self.population.mixes)
                            self.engine.update_mix(self.population.mixes[idx])
                    except Exception as e:
                        self.status_msg = f"Delete Error: {e}"
            self.confirming_delete = False
            return

        if self.mode == "FILE_PICKER":
            # ... existing file picker logic ...
            if self.file_list:
                if self.preview_stream:
                    self.preview_stream.stop()
                
                item = self.file_list[self.selected_file_idx]
                if item["type"] == "dir":
                    self.current_path = os.path.abspath(os.path.join(self.current_path, item["name"]))
                    self.refresh_file_list()
                else:
                    if item["name"].endswith(".json"):
                        try:
                            filepath = os.path.join(self.current_path, item["name"])
                            loaded_mix = PresetManager.load_preset(filepath)
                            mix_idx = self.selected_mix_idx % len(self.population.mixes)
                            self.population.mixes[mix_idx] = loaded_mix
                            self.selected_module_idx = 0
                            self.selected_param_idx = 0
                            self.mode = "EVOLUTION"
                            self.status_msg = f"Loaded preset: {item['name']}"
                            if self.is_playing:
                                self.engine.update_mix(loaded_mix)
                        except Exception as e:
                            self.status_msg = f"Load Failed: {e}"
                    else:
                        self.load_audio(item["name"])
            return

        if self.focus_zone == FocusZone.MENU:
            # ... existing menu logic ...
            item = self.header_menu.items[self.header_menu.selected_index]
            
            if self.in_submenu:
                # Execute the selected child action
                if 0 <= self.selected_child_idx < len(item.children):
                    child = item.children[self.selected_child_idx]
                    if child.action:
                        child.action()
                        self.in_submenu = False
                        # Note: Most actions (Swap, Load) change mode/zone themselves, 
                        # but we fallback to BANDS for consistency if they don't.
                        if self.mode == "EVOLUTION":
                            self.focus_zone = FocusZone.BANDS
            else:
                # Enter sub-menu or execute parent action
                if item.action:
                    item.action()
                    self.focus_zone = FocusZone.BANDS
                elif item.children:
                    self.in_submenu = True
                    self.selected_child_idx = 0
                    self.status_msg = f"Navigating {item.title}... (Arrows: Select, ENTER: Execute, ESC: Back)"
            return

        if self.focus_zone == FocusZone.POPULATION:
            # Evolve selected parent
            idx = self.selected_mix_idx % len(self.population.mixes)
            if self.population.mixes[idx].is_locked:
                self.status_msg = "Cannot evolve a locked mix. Unlock it first (Press K)."
                return
                
            self.population.generate_next_generation(self.selected_mix_idx, 0.5, 0.5)
            if self.is_playing:
                self.engine.update_mix(self.population.mixes[idx])
            return

        if self.focus_zone in (FocusZone.BANDS, FocusZone.STEMS):
            # Use correct cursors
            cur_col_idx = self.stem_col_idx if self.focus_zone == FocusZone.STEMS else self.selected_band_idx
            cur_mod_idx = self.stem_mod_idx if self.focus_zone == FocusZone.STEMS else self.selected_module_idx

            # Handle confirming FX selection
            if self.selecting_new_fx:
                if 0 <= cur_mod_idx < len(self.available_fx_pool):
                    cols = self.get_column_data(1)
                    band = cols[cur_col_idx]
                    if isinstance(band, Band):
                        selected_cls = self.available_fx_pool[cur_mod_idx]
                        new_mod = selected_cls()
                        band.modules.append(new_mod)
                        self.status_msg = f"Added {new_mod.name} to {band.name}"
                        if self.is_playing:
                            idx = self.selected_mix_idx % len(self.population.mixes)
                            self.engine.update_mix(self.population.mixes[idx])
                
                self.selecting_new_fx = False
                if self.focus_zone == FocusZone.STEMS:
                    self.stem_mod_idx = 0
                else:
                    self.selected_module_idx = 0
                return

            if self.focus_zone == FocusZone.BANDS and self.editing_crossover:
                p = self.get_selected_param()
                if p:
                    self.editing = True
                    self.edit_buffer = str(round(p.current_value, 2))
                return

            modules_list = self.get_column_data(2)
            if 0 <= cur_mod_idx < len(modules_list):
                item = modules_list[cur_mod_idx]
                
                # Enter selection mode when [+] Add FX button is clicked
                if item == "[+] Add FX":
                    cols = self.get_column_data(1)
                    band = cols[cur_col_idx]
                    if isinstance(band, Band):
                        # Pool of all available modules (classes)
                        module_pool = [
                            CompressorModule, ExpanderModule, TransientShaperModule, 
                            SaturationModule, ClipperModule, LimiterModule, ConvolutionModule
                        ]
                        existing_classes = {type(m) for m in band.modules}
                        available = [cls for cls in module_pool if cls not in existing_classes]
                        
                        if available:
                            self.selecting_new_fx = True
                            self.available_fx_pool = available
                            if self.focus_zone == FocusZone.STEMS:
                                self.stem_mod_idx = 0
                            else:
                                self.selected_module_idx = 0
                            self.status_msg = "Select effect to add..."
                        else:
                            self.status_msg = "Band is full! Cannot add more FX."
                    return
            
            # Default action: Inline Parameter Editing
            p = self.get_selected_param()
            if p:
                self.editing = True
                self.edit_buffer = str(round(p.current_value, 2))

    def navigate(self, key: str):
        if self.confirming_delete:
            if key == KEY_ENTER:
                self.execute_action()
                return
            elif key == KEY_ESC:
                self.confirming_delete = False
                self.status_msg = "Deletion cancelled."
                return
            else:
                return # Block other inputs during confirmation

        if self.in_export_modal:
            if key == KEY_UP:
                self.export_modal_idx = (self.export_modal_idx - 1) % 4
                return
            elif key == KEY_DOWN:
                self.export_modal_idx = (self.export_modal_idx + 1) % 4
                return
            elif key in (KEY_ENTER, KEY_SPACE):
                if self.export_modal_idx == 0:
                    self.export_opt_full = not self.export_opt_full
                elif self.export_modal_idx == 1:
                    if self.stems_data is not None:
                        self.export_opt_stems = not self.export_opt_stems
                elif self.export_modal_idx == 2:
                    self.in_export_modal = False
                    self.editing = True
                    self.input_mode = "EXPORT_PATH_PROMPT"
                    self.edit_buffer = ""
                    self.status_msg = "Enter save path/name (e.g., my_mix): █"
                elif self.export_modal_idx == 3:
                    self.in_export_modal = False
                    self.status_msg = "Export cancelled."
                return
            elif key == KEY_ESC:
                self.in_export_modal = False
                self.status_msg = "Export cancelled."
                return
            return # Block other inputs

        if self.editing:
            if key == KEY_ENTER:
                if self.input_mode == "SAVE_PRESET":
                    try:
                        mix_idx = self.selected_mix_idx % len(self.population.mixes)
                        mix = self.population.mixes[mix_idx]
                        os.makedirs("./presets", exist_ok=True)
                        filepath = f"./presets/{self.edit_buffer}.json"
                        PresetManager.save_preset(mix, filepath)
                        self.status_msg = f"Saved preset to {filepath}"
                    except Exception as e:
                        self.status_msg = f"Save Failed: {e}"
                elif self.input_mode == "EXPORT_PATH_PROMPT":
                    try:
                        self.status_msg = "Exporting..."
                        mix_idx = self.selected_mix_idx % len(self.population.mixes)
                        mix = self.population.mixes[mix_idx]
                        
                        destination = self.edit_buffer
                        is_full = self.export_opt_full
                        is_stems = self.export_opt_stems
                        
                        export_base_dir = os.path.join(self.output_dir, "output")
                        os.makedirs(export_base_dir, exist_ok=True)
                        
                        if is_full and not is_stems:
                            source = self.stems_data if self.stems_data is not None else self.audio_data
                            exported_audio = self.engine.process(source, mix)
                            exported_audio = np.clip(exported_audio, -1.0, 1.0)
                            output_path = os.path.join(export_base_dir, f"{destination}.wav")
                            sf.write(output_path, exported_audio.T, self.sample_rate)
                            self.status_msg = f"Exported Full Track to {destination}.wav"
                            
                        elif is_stems and not is_full:
                            dest_dir = os.path.join(export_base_dir, destination)
                            stems_dir = os.path.join(dest_dir, "stems")
                            os.makedirs(stems_dir, exist_ok=True)
                            
                            for stem_name, stem_audio in self.stems_data.items():
                                stem_path = os.path.join(stems_dir, f"{destination}_{stem_name}.wav")
                                sf.write(stem_path, stem_audio.T, self.sample_rate)
                                
                            self.status_msg = f"Exported Stems to {stems_dir}/"
                            
                        elif is_full and is_stems:
                            dest_dir = os.path.join(export_base_dir, destination)
                            stems_dir = os.path.join(dest_dir, "stems")
                            os.makedirs(stems_dir, exist_ok=True)
                            
                            # Export Full
                            source = self.stems_data if self.stems_data is not None else self.audio_data
                            exported_audio = self.engine.process(source, mix)
                            exported_audio = np.clip(exported_audio, -1.0, 1.0)
                            full_path = os.path.join(dest_dir, f"{destination}_full_mix.wav")
                            sf.write(full_path, exported_audio.T, self.sample_rate)
                            
                            # Export Stems
                            if self.stems_data is not None:
                                for stem_name, stem_audio in self.stems_data.items():
                                    stem_path = os.path.join(stems_dir, f"{destination}_{stem_name}.wav")
                                    sf.write(stem_path, stem_audio.T, self.sample_rate)
                                
                            self.status_msg = f"Exported Full Track & Stems to {dest_dir}/"
                            
                        else:
                            self.status_msg = "Nothing selected for export."
                    except Exception as e:
                        self.status_msg = f"Export Failed: {e}"
                else:
                    try: self.set_value(float(self.edit_buffer))
                    except: pass
                self.editing = False; self.edit_buffer = ""; self.input_mode = None
            elif key == KEY_ESC:
                self.editing = False; self.edit_buffer = ""; self.input_mode = None
                self.status_msg = "Cancelled."
            elif key == KEY_BACKSPACE:
                self.edit_buffer = self.edit_buffer[:-1]
            elif key and len(key) == 1 and key != " ":
                self.edit_buffer += key
            return

        if self.selecting_new_fx:
            if key == KEY_ESC:
                self.selecting_new_fx = False
                self.status_msg = "Cancelled FX addition."
                return
            elif key == KEY_ENTER:
                self.execute_action()
                return

        if key == KEY_TAB and self.mode == "EVOLUTION":
            zones = [FocusZone.MENU, FocusZone.POPULATION, FocusZone.BANDS, FocusZone.STEMS, FocusZone.PLAYBACK]
            idx = zones.index(self.focus_zone)
            
            # Skip STEMS zone if separating
            while True:
                idx = (idx + 1) % len(zones)
                self.focus_zone = zones[idx]
                if self.focus_zone == FocusZone.STEMS and self.is_separating:
                    continue # Skip STEMS if separating
                break
                
            self.selecting_new_fx = False # Exit selection mode when switching zones
            self.confirming_delete = False
            return

        if key == KEY_SPACE:
            self.toggle_playback()
            return

        if key == KEY_ENTER:
            self.execute_action()
            return

        # Delete FX Logic
        if key == KEY_D and self.focus_zone in (FocusZone.BANDS, FocusZone.STEMS) and not self.editing:
            modules_list = self.get_column_data(2)
            cur_mod_idx = self.stem_mod_idx if self.focus_zone == FocusZone.STEMS else self.selected_module_idx
            if 0 <= cur_mod_idx < len(modules_list):
                item = modules_list[cur_mod_idx]
                if not isinstance(item, str) and not isinstance(item, Parameter):
                    # It's an AudioModule
                    self.confirming_delete = True
                    self.status_msg = f"Delete {item.name}? (ENTER: Yes, ESC: No)"
                    return

        # Mix Sorting logic (Depth 0 = FocusZone.POPULATION)
        # ... (mix sorting logic unchanged)
        if self.focus_zone == FocusZone.POPULATION:
            if key == KEY_K:
                idx = self.selected_mix_idx % len(self.population.mixes)
                self.population.mixes[idx].is_locked = not self.population.mixes[idx].is_locked
                status = "Locked" if self.population.mixes[idx].is_locked else "Unlocked"
                self.status_msg = f"Mix {idx} {status}."
                # Allow fallthrough to _handle_population_input if we just locked, but we shouldn't return here if we want left/right to keep working.
                # Actually, locking shouldn't break navigation unless we return early. Let's just not return here!
            elif key == KEY_SHIFT_UP:
                idx = self.selected_mix_idx % len(self.population.mixes)
                prev_idx = (idx - 1) % len(self.population.mixes)
                self.population.mixes[idx], self.population.mixes[prev_idx] = self.population.mixes[prev_idx], self.population.mixes[idx]
                self.selected_mix_idx = prev_idx
                return
            elif key == KEY_SHIFT_DOWN:
                idx = self.selected_mix_idx % len(self.population.mixes)
                next_idx = (idx + 1) % len(self.population.mixes)
                self.population.mixes[idx], self.population.mixes[next_idx] = self.population.mixes[next_idx], self.population.mixes[idx]
                self.selected_mix_idx = next_idx
                return

        # If in FILE_PICKER, handle remaining inputs
        if self.mode == "FILE_PICKER":
            # ... (file picker logic unchanged)
            if key == KEY_UP:
                self.selected_file_idx = (self.selected_file_idx - 1) % max(1, len(self.file_list))
                self._preview_selected_file()
            elif key == KEY_DOWN:
                self.selected_file_idx = (self.selected_file_idx + 1) % max(1, len(self.file_list))
                self._preview_selected_file()
            elif key == KEY_BACKSPACE:
                if self.preview_stream:
                    self.preview_stream.stop()
                self.current_path = os.path.abspath(os.path.join(self.current_path, ".."))
                self.refresh_file_list()
            elif key == KEY_ESC:
                if self.preview_stream:
                    self.preview_stream.stop()
                if self.audio_data is not None:
                    self.mode = "EVOLUTION"
                    self.status_msg = "Evolution Mode."
            return

        # Evolution Global Shortcuts
        if key == KEY_L:
            self.mode = "FILE_PICKER"
            self.refresh_file_list()
            return
        elif key == KEY_BRACKET_LEFT:
            if self.focus_zone == FocusZone.STEMS:
                self.stem_col_idx = (self.stem_col_idx - 1) % len(self.get_column_data(1))
                self.stem_mod_idx = 0; self.stem_param_idx = 0
            else:
                self.selected_band_idx = (self.selected_band_idx - 1) % len(self.get_column_data(1))
                self.selected_module_idx = 0; self.selected_param_idx = 0; self.editing_crossover = False
            self.selecting_new_fx = False
        elif key == KEY_BRACKET_RIGHT:
            if self.focus_zone == FocusZone.STEMS:
                self.stem_col_idx = (self.stem_col_idx + 1) % len(self.get_column_data(1))
                self.stem_mod_idx = 0; self.stem_param_idx = 0
            else:
                self.selected_band_idx = (self.selected_band_idx + 1) % len(self.get_column_data(1))
                self.selected_module_idx = 0; self.selected_param_idx = 0; self.editing_crossover = False
            self.selecting_new_fx = False
        elif key == KEY_M:
            cols = self.get_column_data(1)
            cur_col_idx = self.stem_col_idx if self.focus_zone == FocusZone.STEMS else self.selected_band_idx
            band = cols[cur_col_idx]
            if isinstance(band, Band):
                band.is_muted = not band.is_muted
                self.status_msg = f"{band.name} {'Muted' if band.is_muted else 'Unmuted'}"
                if self.is_playing:
                    idx = self.selected_mix_idx % len(self.population.mixes)
                    self.engine.update_mix(self.population.mixes[idx])
        elif key == KEY_S:
            cols = self.get_column_data(1)
            cur_col_idx = self.stem_col_idx if self.focus_zone == FocusZone.STEMS else self.selected_band_idx
            band = cols[cur_col_idx]
            if isinstance(band, Band):
                band.is_soloed = not band.is_soloed
                self.status_msg = f"{band.name} {'Soloed' if band.is_soloed else 'Unsoloed'}"
                if self.is_playing:
                    idx = self.selected_mix_idx % len(self.population.mixes)
                    self.engine.update_mix(self.population.mixes[idx])
        elif key == KEY_K:
            if self.focus_zone != FocusZone.POPULATION:
                p = self.get_selected_param()
                if p:
                    p.is_locked = not p.is_locked
                    self.status_msg = f"Param {p.name} {'Locked' if p.is_locked else 'Unlocked'}"

        if self.focus_zone == FocusZone.MENU:
            self._handle_menu_input(key)
        elif self.focus_zone == FocusZone.POPULATION:
            self._handle_population_input(key)
        elif self.focus_zone == FocusZone.BANDS:
            self._handle_bands_input(key)
        elif self.focus_zone == FocusZone.PLAYBACK:
            self._handle_playback_input(key)
        elif self.focus_zone == FocusZone.STEMS:
            self._handle_stems_input(key)

    def _handle_stems_input(self, key: str):
        if self.is_separating:
            self.status_msg = "Cannot adjust stems while separating..."
            return
            
        if not self.stems_data:
            self.status_msg = "No stems available. Separate stems first."
            return

        cols = self.get_column_data(1)
        if self.stem_col_idx >= len(cols): self.stem_col_idx = 0
        current_col = cols[self.stem_col_idx]
        
        # If selecting new FX, restrict navigation to the pool list
        if self.selecting_new_fx:
            if key == KEY_UP:
                self.stem_mod_idx = (self.stem_mod_idx - 1) % len(self.available_fx_pool)
            elif key == KEY_DOWN:
                self.stem_mod_idx = (self.stem_mod_idx + 1) % len(self.available_fx_pool)
            return

        # Special case: Quick compare across columns
        if key == KEY_CTRL_LEFT:
            self.stem_col_idx = (self.stem_col_idx - 1) % len(cols)
            return
        elif key == KEY_CTRL_RIGHT:
            self.stem_col_idx = (self.stem_col_idx + 1) % len(cols)
            return

        if key == KEY_UP:
            if self.stem_param_idx > 0:
                self.stem_param_idx -= 1
            elif self.stem_mod_idx > 0:
                self.stem_mod_idx -= 1
                current_depth2 = self.get_column_data(2)
                prev_mod = current_depth2[self.stem_mod_idx]
                if hasattr(prev_mod, "parameters"):
                    self.stem_param_idx = len(prev_mod.parameters) - 1
                else:
                    self.stem_param_idx = 0
            elif self.stem_mod_idx == 0:
                self.stem_mod_idx = -1
                self.stem_param_idx = 0
                
        elif key == KEY_CTRL_UP:
            if self.stem_mod_idx > 0:
                self.stem_mod_idx -= 1
                self.stem_param_idx = 0
            elif self.stem_mod_idx == 0:
                self.stem_mod_idx = -1
                
        elif key == KEY_DOWN:
            current_depth2 = self.get_column_data(2)
            if self.stem_mod_idx == -1:
                if len(current_depth2) > 0:
                    self.stem_mod_idx = 0
                    self.stem_param_idx = 0
                return

            if 0 <= self.stem_mod_idx < len(current_depth2):
                mod = current_depth2[self.stem_mod_idx]
                num_params = len(mod.parameters) if hasattr(mod, "parameters") else 0
                
                if self.stem_param_idx < num_params - 1:
                    self.stem_param_idx += 1
                elif self.stem_mod_idx < len(current_depth2) - 1:
                    self.stem_mod_idx += 1
                    self.stem_param_idx = 0
                    
        elif key == KEY_CTRL_DOWN:
            current_depth2 = self.get_column_data(2)
            if self.stem_mod_idx == -1:
                if current_depth2:
                    self.stem_mod_idx = 0
                    self.stem_param_idx = 0
                return

            if self.stem_mod_idx < len(current_depth2) - 1:
                self.stem_mod_idx += 1
                self.stem_param_idx = 0
        
        elif key == KEY_LEFT:
            self.adjust_value(-1)
        elif key == KEY_RIGHT:
            self.adjust_value(1)
        elif key == KEY_B:
            current_col.is_bypassed = not current_col.is_bypassed
            self.status_msg = f"{current_col.name} {'Bypassed' if current_col.is_bypassed else 'Active'}"
            if self.is_playing:
                self.engine.update_mix(self.population.mixes[self.selected_mix_idx % len(self.population.mixes)])
        elif key == KEY_SHIFT_UP: # Granular up
            self.adjust_value(1, granular=True)
        elif key == KEY_SHIFT_DOWN: # Granular down
            self.adjust_value(-1, granular=True)

    def _handle_menu_input(self, key: str):
        if not self.header_menu.items: return
        
        item = self.header_menu.items[self.header_menu.selected_index]
        
        if self.in_submenu:
            if key == KEY_LEFT:
                self.selected_child_idx = (self.selected_child_idx - 1) % len(item.children)
            elif key == KEY_RIGHT:
                self.selected_child_idx = (self.selected_child_idx + 1) % len(item.children)
            elif key == KEY_UP: # Allow vertical navigation to exit sub-menu
                self.in_submenu = False
            elif key == KEY_ESC:
                self.in_submenu = False
            return

        if key == KEY_LEFT:
            self.header_menu.selected_index = (self.header_menu.selected_index - 1) % len(self.header_menu.items)
        elif key == KEY_RIGHT:
            self.header_menu.selected_index = (self.header_menu.selected_index + 1) % len(self.header_menu.items)

    def _handle_playback_input(self, key: str):
        if key in (KEY_LEFT, KEY_CTRL_LEFT):
            if self.sample_rate:
                amount = 1 if key == KEY_CTRL_LEFT else 5
                # Scrub back safely
                self.play_idx = max(0, self.play_idx - (amount * self.sample_rate))
        elif key in (KEY_RIGHT, KEY_CTRL_RIGHT):
            if self.sample_rate and self.audio_data is not None:
                amount = 1 if key == KEY_CTRL_RIGHT else 5
                # Scrub forward safely
                self.play_idx = min(self.audio_data.shape[1] - 1, self.play_idx + (amount * self.sample_rate))
        elif key == KEY_B:
            # Toggle global bypass for the current mix
            mix = self.population.mixes[self.selected_mix_idx % len(self.population.mixes)]
            all_bands = [mix.pre_band, mix.post_band] + mix.bands
            if mix.stem_bands:
                all_bands += list(mix.stem_bands.values())
            
            # If any band is NOT bypassed, bypass all. Otherwise un-bypass all.
            target_state = not all(b.is_bypassed for b in all_bands)
            for b in all_bands:
                b.is_bypassed = target_state
            
            self.status_msg = f"All bands {'Bypassed' if target_state else 'Active'}"
            if self.is_playing:
                self.engine.update_mix(mix)

    def _handle_population_input(self, key: str):
        if key == KEY_LEFT:
            self.selected_mix_idx = (self.selected_mix_idx - 1) % len(self.population.mixes)
            if self.is_playing:
                self.engine.update_mix(self.population.mixes[self.selected_mix_idx])
        elif key == KEY_RIGHT:
            self.selected_mix_idx = (self.selected_mix_idx + 1) % len(self.population.mixes)
            if self.is_playing:
                self.engine.update_mix(self.population.mixes[self.selected_mix_idx])

    def _handle_bands_input(self, key: str):
        mix = self.population.mixes[self.selected_mix_idx % len(self.population.mixes)]
        
        cols = self.get_column_data(1)
        if self.selected_band_idx >= len(cols): self.selected_band_idx = 0
        current_col = cols[self.selected_band_idx]
        
        # If selecting new FX, restrict navigation to the pool list
        if self.selecting_new_fx:
            if key == KEY_UP:
                self.selected_module_idx = (self.selected_module_idx - 1) % len(self.available_fx_pool)
            elif key == KEY_DOWN:
                self.selected_module_idx = (self.selected_module_idx + 1) % len(self.available_fx_pool)
            return

        # Special case: Quick compare across columns
        if key == KEY_CTRL_LEFT:
            self.selected_band_idx = (self.selected_band_idx - 1) % len(cols)
            self.editing_crossover = False
            return
        elif key == KEY_CTRL_RIGHT:
            self.selected_band_idx = (self.selected_band_idx + 1) % len(cols)
            self.editing_crossover = False
            return

        if key == KEY_UP:
            if self.editing_crossover:
                return # Topmost element
                
            if self.selected_param_idx > 0:
                self.selected_param_idx -= 1
            elif self.selected_module_idx > 0:
                self.selected_module_idx -= 1
                # Find number of parameters for previous module
                current_depth2 = self.get_column_data(2)
                prev_mod = current_depth2[self.selected_module_idx]
                if hasattr(prev_mod, "parameters"):
                    self.selected_param_idx = len(prev_mod.parameters) - 1
                else:
                    self.selected_param_idx = 0
            elif self.selected_module_idx == 0:
                # Move to Band Gain
                self.selected_module_idx = -1
                self.selected_param_idx = 0
            elif self.selected_module_idx == -1:
                # Move to Crossover Title (if on a frequency band)
                if 1 <= self.selected_band_idx <= len(mix.crossover_params):
                    self.editing_crossover = True
        elif key == KEY_CTRL_UP:
            if self.editing_crossover:
                return
            if self.selected_module_idx > 0:
                self.selected_module_idx -= 1
                self.selected_param_idx = 0
            elif self.selected_module_idx == 0:
                self.selected_module_idx = -1
            elif self.selected_module_idx == -1:
                if 1 <= self.selected_band_idx <= len(mix.crossover_params):
                    self.editing_crossover = True
        elif key == KEY_DOWN:
            if self.editing_crossover:
                self.editing_crossover = False
                self.selected_module_idx = -1
                self.selected_param_idx = 0
                return
                
            current_depth2 = self.get_column_data(2)
            if self.selected_module_idx == -1:
                if current_depth2:
                    self.selected_module_idx = 0
                    self.selected_param_idx = 0
                return

            if 0 <= self.selected_module_idx < len(current_depth2):
                mod = current_depth2[self.selected_module_idx]
                if hasattr(mod, "parameters") and self.selected_param_idx < len(mod.parameters) - 1:
                    self.selected_param_idx += 1
                elif self.selected_module_idx < len(current_depth2) - 1:
                    self.selected_module_idx += 1
                    self.selected_param_idx = 0
        elif key == KEY_CTRL_DOWN:
            if self.editing_crossover:
                self.editing_crossover = False
                self.selected_module_idx = -1
                return

            current_depth2 = self.get_column_data(2)
            if self.selected_module_idx == -1:
                if current_depth2:
                    self.selected_module_idx = 0
                    self.selected_param_idx = 0
                return

            if self.selected_module_idx < len(current_depth2) - 1:
                self.selected_module_idx += 1
                self.selected_param_idx = 0
        elif key == KEY_LEFT:
            self.adjust_value(-1)
        elif key == KEY_RIGHT:
            self.adjust_value(1)
        elif key == KEY_B:
            current_col.is_bypassed = not current_col.is_bypassed
            self.status_msg = f"{current_col.name} {'Bypassed' if current_col.is_bypassed else 'Active'}"
            if self.is_playing:
                self.engine.update_mix(mix)
        elif key == KEY_CTRL_LEFT: # Redundant but safe
            self.adjust_value(-1, granular=True)
        elif key == KEY_CTRL_RIGHT: # Redundant but safe
            self.adjust_value(1, granular=True)

    def render(self):
        # Build the UI Layout dynamically to avoid out-of-bounds rendering
        layout = Layout()
        
        # Common Header and Status
        menu_texts = []
        for i, item in enumerate(self.header_menu.items):
            is_parent_selected = (self.focus_zone == FocusZone.MENU and i == self.header_menu.selected_index)
            style = "bold white on blue" if is_parent_selected else "white"
            menu_texts.append(Text(f" [{item.title}] ", style=style))
            
            # If this is the active sub-menu, render children
            if is_parent_selected and self.in_submenu:
                for j, child in enumerate(item.children):
                    is_child_sel = (j == self.selected_child_idx)
                    child_style = "bold black on yellow" if is_child_sel else "cyan"
                    menu_texts.append(Text(f" {child.title} ", style=child_style))
            
        header_left = Text(f" GAIA | {self.metadata.filename or '---'}", style="bold magenta")
        header_gen = Text(f"Gen {self.population.generation_count}", style="bold cyan")
        header_content = Columns([header_left, header_gen] + menu_texts, expand=True)
        header_panel = Panel(header_content, box=box.DOUBLE, border_style="magenta" if self.focus_zone == FocusZone.MENU else "dim")
        
        # Footer / Status Panel
        meta = self.metadata
        bpm_str = f"{meta.bpm} BPM" if meta.bpm else "No BPM"
        meta_str = f"File: {meta.filename or 'None'} | {meta.sample_rate}Hz | {meta.channels}ch | Block: {self.blocksize} | {bpm_str}"
        
        display_msg = self.status_msg
        if self.editing and "█" in self.status_msg:
            display_msg = self.status_msg.replace("█", self.edit_buffer + "█")
            
        footer_text = f"{display_msg}  |  {meta_str}"
        status_panel = Panel(Text(footer_text, style="bold yellow" if self.editing else "italic green"), box=box.SIMPLE)

        if self.in_export_modal:
            lines = []
            
            style0 = "bold black on white" if self.export_modal_idx == 0 else "white"
            mark0 = "[X]" if self.export_opt_full else "[ ]"
            lines.append(Text(f" {mark0} Full Track ", style=style0))
            
            style1 = "bold black on white" if self.export_modal_idx == 1 else "white"
            mark1 = "[X]" if self.export_opt_stems else "[ ]"
            if self.stems_data is None:
                style1 = "dim" if self.export_modal_idx != 1 else "bold black on grey50"
                lines.append(Text(f" [ ] Stems (Unavailable) ", style=style1))
            else:
                lines.append(Text(f" {mark1} Stems (Vocals, Drums, Bass, Other) ", style=style1))
                
            style2 = "bold black on white" if self.export_modal_idx == 2 else "green"
            lines.append(Text(" [ Continue ] ", style=style2))
            
            style3 = "bold black on white" if self.export_modal_idx == 3 else "red"
            lines.append(Text(" [ Cancel ] ", style=style3))
            
            modal_content = Group(*[Align.center(line) for line in lines])
            modal_panel = Panel(
                modal_content,
                title="Export Settings",
                border_style="magenta",
                box=box.DOUBLE,
                padding=(1, 4)
            )
            
            layout.split_column(
                Layout(header_panel, size=3),
                Layout(Align.center(modal_panel, vertical="middle"), ratio=1),
                Layout(status_panel, size=3)
            )
            return layout

        if self.mode == "FILE_PICKER":
            # File Browser Layout
            try:
                term_lines = os.get_terminal_size().lines
                browser_height = max(5, term_lines - 10)
            except:
                browser_height = 10
                
            start_idx = max(0, self.selected_file_idx - browser_height // 2)
            visible_files = self.file_list[start_idx:start_idx + browser_height]
            
            content = [Text(f"Path: {self.current_path}", style="bold yellow"), Text("-" * 20, style="dim")]
            for i, item in enumerate(visible_files):
                idx = start_idx + i
                prefix = "▶ " if idx == self.selected_file_idx else "  "
                style = "bold white on blue" if idx == self.selected_file_idx else "white"
                display_name = f"[DIR] {item['name']}" if item["type"] == "dir" else item["name"]
                content.append(Text(f"{prefix}{display_name}", style=style))
                
            browser_panel = Panel(Group(*content), title="File Browser", border_style="cyan")
            
            layout.split_column(
                Layout(header_panel, size=3),
                Layout(browser_panel, ratio=1),
                Layout(status_panel, size=3)
            )
            return layout

        else:
            # Evolution Layout
            
            # 1. Population Row
            mix_texts = []
            for i in range(len(self.population.mixes)):
                is_sel = (self.focus_zone == FocusZone.POPULATION and i == self.selected_mix_idx)
                prefix = "▶ " if is_sel else "  "
                lock_indicator = " [*]" if self.population.mixes[i].is_locked else ""
                style = "bold white on blue" if is_sel else "white" if i == self.selected_mix_idx else "dim"
                mix_texts.append(Text(f"{prefix}Mix {i}{lock_indicator}", style=style))
                
            pop_panel = Panel(Columns(mix_texts, expand=True, align="center"), border_style="magenta" if (self.focus_zone == FocusZone.POPULATION) else "dim")

            # 2. Bands Row
            idx = self.selected_mix_idx % len(self.population.mixes)
            mix = self.population.mixes[idx]
            
            # Use explicit band list for layout, NOT get_column_data(1) which can switch to Stems
            band_columns = [mix.pre_band] + mix.bands + [mix.post_band]
            band_layout = Layout(name="bands")
            band_layout.split_row(*[Layout(name=f"col_{i}") for i in range(len(band_columns))])
            
            for i, band in enumerate(band_columns):
                is_col_focused = (self.focus_zone == FocusZone.BANDS and self.selected_band_idx == i)
                
                # Rendering a Band (PRE, POST, or Frequency Band)
                items = []
                
                # Mute / Solo / Bypass indicators
                ms_text = Text("")
                if band.is_soloed: ms_text.append("[S]", style="bold black on yellow")
                if band.is_muted: ms_text.append("[M]", style="bold white on red")
                if getattr(band, 'is_bypassed', False): ms_text.append("[B]", style="bold white on blue")
                if len(ms_text) > 0: items.append(ms_text)

                # Band Title & Crossover logic
                is_xo_sel = is_col_focused and self.editing_crossover
                if 1 <= i <= len(mix.crossover_params):
                    # This is a frequency band (Low, Mid, etc.) with a Crossover
                    xo = mix.crossover_params[i-1]
                    txt = self.edit_buffer if (is_xo_sel and self.editing) else f"{xo.current_value:.1f}"
                    if xo.is_locked: txt = f"*{txt}"
                    band_title = Text(f"{band.name} | {txt}Hz", style="bold yellow" if is_xo_sel else "cyan")
                else:
                    # PRE, POST, or the last frequency band (High/Nyq)
                    band_title = Text(band.name, style="bold magenta" if (band.name in ["PRE", "POST"]) else "cyan")

                # Band Gain (Subtitle)
                is_gain_sel = is_col_focused and self.selected_module_idx == -1 and not self.editing_crossover
                gain_val = self.edit_buffer if (is_gain_sel and self.editing) else f"{band.gain.current_value:.1f}"
                if band.gain.is_locked: gain_val = f"*{gain_val}"
                gain_style = "bold black on yellow" if (is_gain_sel and self.editing) else "bold white on blue" if is_gain_sel else "white"
                band_subtitle = Text(f" {gain_val}dB ", style=gain_style)

                # Modules rendering
                # Only the focused band shows the [+] Add FX button to save space and clarify focus
                if is_col_focused:
                    modules_to_render = self.get_column_data(2)
                else:
                    modules_to_render = band.modules
                
                items.append(self.render_column_content(modules_to_render, 2, is_focused=is_col_focused))
                    
                band_panel = Panel(
                    Group(*items), 
                    title=band_title,
                    title_align="right",
                    subtitle=band_subtitle,
                    subtitle_align="center",
                    border_style="blue" if (is_col_focused and not self.editing_crossover) else "yellow" if is_xo_sel else "dim"
                )
                band_layout[f"col_{i}"].update(band_panel)

            # 3. Audio Progress Row
            idx = self.selected_mix_idx % len(self.population.mixes)
            mix = self.population.mixes[idx]

            # Check if all relevant bands are bypassed
            all_b = [mix.pre_band, mix.post_band] + mix.bands
            if mix.stem_bands: all_b += list(mix.stem_bands.values())
            is_global_bypass = all(getattr(b, 'is_bypassed', False) for b in all_b)
            bypass_str = " (BYPASSED)" if is_global_bypass else ""

            elapsed = self.play_idx / self.sample_rate if self.sample_rate else 0
            pct = (elapsed / self.playback_duration) if self.playback_duration > 0 else 0
            bars_filled = int(pct * 20)
            bars_empty = 20 - bars_filled
            progress_bar = f"[{'█' * bars_filled}{'░' * bars_empty}]"

            if self.is_playing:
                prog_str = f"{progress_bar} {elapsed:.1f}s / {self.playback_duration:.1f}s{bypass_str}"
                audio_panel = Panel(Text(prog_str, style="bold cyan" if not is_global_bypass else "bold white on blue"), title="Audio", border_style="green" if self.focus_zone == FocusZone.PLAYBACK else "dim")
            else:
                prog_str = f"{progress_bar} {elapsed:.1f}s / {self.playback_duration:.1f}s (Stopped){bypass_str}"
                audio_panel = Panel(Text(prog_str, style="dim" if not is_global_bypass else "bold white on blue"), title="Audio", border_style="cyan" if self.focus_zone == FocusZone.PLAYBACK else "dim")
            # 3.5 Stems Row (Miller Columns)
            stems_layout = Layout(name="stems")
            if self.is_separating:
                bars_filled = int((self.separation_progress / 100.0) * 40)
                bars_empty = 40 - bars_filled
                p_bar = f"[{'█' * bars_filled}{'░' * bars_empty}]"
                stem_panel = Panel(Text(f" Separating Stems... {p_bar} {self.separation_progress:.1f}% ", style="bold black on yellow"), border_style="yellow")
                stems_layout.update(stem_panel)
            elif not self.stems_data:
                stem_panel = Panel(Text(" No Stems (Use Stems -> Separate Stems) ", style="dim"), border_style="dim")
                stems_layout.update(stem_panel)
            else:
                stems_layout.split_row(*[Layout(name=f"stem_col_{i}") for i in range(len(self.stem_names))])
                idx = self.selected_mix_idx % len(self.population.mixes)
                mix = self.population.mixes[idx]
                
                for i, stem_name in enumerate(self.stem_names):
                    is_col_focused = (self.focus_zone == FocusZone.STEMS and self.stem_col_idx == i)
                    band = mix.stem_bands[stem_name]
                    items = []
                    
                    # Mute / Solo / Bypass indicators
                    ms_text = Text("")
                    if band.is_soloed: ms_text.append("[S]", style="bold black on yellow")
                    if band.is_muted: ms_text.append("[M]", style="bold white on red")
                    if getattr(band, 'is_bypassed', False): ms_text.append("[B]", style="bold white on blue")
                    if len(ms_text) > 0: items.append(ms_text)

                    # Title: Name and Gain
                    is_gain_sel = is_col_focused and self.stem_mod_idx == -1
                    gain_val = self.edit_buffer if (is_gain_sel and self.editing) else f"{band.gain.current_value:.1f}"
                    if band.gain.is_locked: gain_val = f"*{gain_val}"
                    gain_style = "bold black on yellow" if (is_gain_sel and self.editing) else "bold white on blue" if is_gain_sel else "white"
                    stem_title = Text(f" {stem_name.upper()} | {gain_val}dB ", style=gain_style)

                    # Modules
                    if is_col_focused:
                        modules_to_render = self.get_column_data(2)
                    else:
                        modules_to_render = band.modules
                    
                    items.append(self.render_column_content(modules_to_render, 2, is_focused=is_col_focused))
                    
                    stem_panel = Panel(
                        Group(*items), 
                        title=stem_title,
                        title_align="center",
                        border_style="blue" if is_col_focused else "dim"
                    )
                    stems_layout[f"stem_col_{i}"].update(stem_panel)

            # Final Assembly of Evolution mode
            layout.split_column(
                Layout(header_panel, size=3),
                Layout(pop_panel, size=3),
                Layout(stems_layout, ratio=1),
                band_layout, # takes remaining ratio=1
                Layout(audio_panel, size=3),
                Layout(status_panel, size=3)
            )
            return layout

def main():
    parser = argparse.ArgumentParser(description="GAIA TUI")
    parser.add_argument("input_file", type=str, nargs="?", help="Optional audio file to load on start")
    parser.add_argument("--blocksize", type=int, default=512, help="Audio buffer block size (default: 512)")
    args = parser.parse_args()

    # Initialize a dummy population
    initial_mixes = [Mix(crossovers=[100.0, 500.0, 2500.0, 8000.0]) for _ in range(5)]
    
    # Lock the first mix by default so user has a safe workspace
    initial_mixes[0].is_locked = True

    pop = Population(initial_mixes)
    tui = GaiaTUI(pop, None, 44100, ".", blocksize=args.blocksize)

    if args.input_file:
        tui.load_audio(args.input_file)

    with TerminalMode() as term:
        # Use Rich Live to render the UI. Set auto_refresh=False to avoid background thread issues.
        with Live(screen=True, auto_refresh=False) as live:
            last_render_time = 0
            while tui.running:
                try:
                    # Throttle render updates to ~15 FPS to prevent terminal overflow/looping
                    current_time = time.time()
                    if current_time - last_render_time > 1.0 / 15.0:
                        live.update(tui.render(), refresh=True)
                        last_render_time = current_time
                    
                    # Check for keyboard input non-blocking
                    key = get_key(term.fd)
                    if key == KEY_CTRL_C:
                        tui.running = False
                    elif key:
                        tui.navigate(key)
                        # Force immediate render on user input
                        live.update(tui.render(), refresh=True)
                        last_render_time = time.time()
                        
                except Exception as e:
                    import traceback
                    with open("crash.log", "w") as f:
                        traceback.print_exc(file=f)
                    tui.running = False
                
                # Prevent CPU spin
                time.sleep(0.01)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
