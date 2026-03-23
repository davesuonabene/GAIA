import sys
import os
import argparse
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
from typing import List, Any, Callable, Optional
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
    def __init__(self, population: Population, audio_data: np.ndarray, sample_rate: int, output_dir: str):
        self.population = population
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        
        # Navigation State
        self.focus_zone = FocusZone.POPULATION
        self.header_menu = HeaderMenu(items=[
            MenuItem(title="File", children=[
                MenuItem(title="Swap Input Track", action=self.action_swap_track),
                MenuItem(title="Load Preset", action=self.action_load_preset_flow),
                MenuItem(title="Save Preset", action=self.action_save_preset_flow),
                MenuItem(title="Export Track", action=self.action_export_track_flow),
            ]),
        ])
        self.metadata = TrackMetadata()
        self.selected_mix_idx = 0
        self.selected_file_idx = 0 # Separate index for file browser
        self.selected_band_idx = 0
        self.selected_module_idx = 0
        self.selected_param_idx = 0
        self.editing_crossover = False
        
        # FX Selection Mode
        self.selecting_new_fx = False
        self.available_fx_pool = []
        
        # Dynamic Input Mode
        self.editing = False
        self.edit_buffer = ""
        self.input_mode = None # "SAVE_PRESET", "EXPORT_TRACK", or None
        
        # Sub-menu State
        self.in_submenu = False
        self.selected_child_idx = 0
        
        self.confirming_delete = False
        
        self.running = True
        self.engine = Engine(sample_rate=self.sample_rate) if self.sample_rate else None
        self.is_playing = False
        self.playback_start_time = 0
        self.playback_duration = 0
        
        # Audio Streaming State
        self.processed_audio = None
        self.stream = None
        self.play_idx = 0
        self.needs_reprocessing = False
        import threading
        self.process_thread = None
        
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

    def reprocess_audio_task(self):
        if self.audio_data is None: return
        try:
            idx = self.selected_mix_idx % len(self.population.mixes)
            mix = self.population.mixes[idx]
            new_audio = self.engine.process(self.audio_data, mix)
            self.processed_audio = new_audio
        except Exception as e:
            pass

    def request_reprocessing(self):
        import threading
        if self.process_thread is None or not self.process_thread.is_alive():
            self.process_thread = threading.Thread(target=self.reprocess_audio_task, daemon=True)
            self.process_thread.start()

    def action_export_track_flow(self):
        """Initiate the export flow."""
        if self.audio_data is None:
            self.status_msg = "Nothing to export."
            return
        self.editing = True
        self.input_mode = "EXPORT_TRACK"
        self.edit_buffer = ""
        self.status_msg = "Enter export filename (no ext): █ (ENTER to export, ESC to cancel)"

    def action_swap_track(self):
        """Reset state and go back to File Picker."""
        if self.is_playing:
            self.toggle_playback()
        self.audio_data = None
        self.engine = None
        self.mode = "FILE_PICKER"
        self.status_msg = "Pick a new audio track."
        self.refresh_file_list()

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

    def audio_callback(self, outdata, frames, time_info, status):
        if self.processed_audio is None:
            outdata.fill(0)
            return
        
        end_idx = min(self.play_idx + frames, self.processed_audio.shape[1])
        n = end_idx - self.play_idx
        if n > 0:
            outdata[:n, :] = self.processed_audio[:, self.play_idx:end_idx].T
        if n < frames:
            outdata[n:, :] = 0
            self.play_idx = 0 # loop playback seamlessly
        else:
            self.play_idx += frames

    def load_audio(self, filename: str):
        if self.is_playing:
            self.toggle_playback() # stop playback safely
        path = os.path.join(self.current_path, filename)
        try:
            data, sr = sf.read(path)
            self.audio_data = data.T if data.ndim > 1 else data.reshape(1, -1)
            self.sample_rate = sr
            self.engine = Engine(sample_rate=self.sample_rate)
            self.mode = "EVOLUTION"
            self.selected_mix_idx = 0 # Reset to prevent IndexError
            self.processed_audio = self.audio_data.copy()
            self.play_idx = 0
            
            # Update Track Metadata
            duration_sec = self.audio_data.shape[1] / sr
            self.metadata.filename = filename
            self.metadata.sample_rate = sr
            self.metadata.channels = self.audio_data.shape[0]
            self.metadata.duration_sec = duration_sec
            
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
                self.status_msg = "Processing..."
                self.reprocess_audio_task() # Initial synchronous process
                self.playback_duration = self.processed_audio.shape[1] / self.sample_rate
                
                # Low-latency settings for Pipewire/Linux
                # A blocksize of 256 or 512 is standard for real-time response
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate, 
                    channels=self.processed_audio.shape[0], 
                    callback=self.audio_callback,
                    blocksize=512
                )
                self.stream.start()
                self.is_playing = True
                self.status_msg = "Playing..."
            except Exception as e:
                self.status_msg = f"Error: {e}"

    def get_column_data(self, depth: int) -> List[Any]:
        """Returns the data list for a specific depth in the navigation tree."""
        idx = self.selected_mix_idx % len(self.population.mixes)
        mix = self.population.mixes[idx]
        
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

            for m_idx, item in enumerate(column_data):
                is_mod_sel = is_focused and (not self.editing_crossover) and self.selected_module_idx == m_idx
                
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
                
                # Render parameters
                for p_idx, (p_name, p) in enumerate(item.parameters.items()):
                    is_p_sel = is_mod_sel and self.selected_param_idx == p_idx
                    val_txt = self.edit_buffer if (is_p_sel and self.editing) else f"{p.current_value:.1f}"
                    if p.is_locked: val_txt = f"*{val_txt}"
                    p_style = "bold black on yellow" if (is_p_sel and self.editing) else "bold white on cyan" if is_p_sel else "dim"
                    items.append(Align.right(Text(f" {p_name}: {val_txt}", style=p_style, overflow="ellipsis", no_wrap=True)))
        
        return Group(*items)

    def get_selected_param(self):
        if self.mode != "EVOLUTION" or self.selecting_new_fx: return None
        idx = self.selected_mix_idx % len(self.population.mixes)
        mix = self.population.mixes[idx]
        
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
        if self.selected_module_idx == -1 and isinstance(parent, Band):
            return parent.gain
            
        if 0 <= self.selected_module_idx < len(modules_list):
            item = modules_list[self.selected_module_idx]
            
            # If it's an AudioModule (Band column)
            if hasattr(item, "parameters"):
                params = list(item.parameters.values())
                return params[self.selected_param_idx] if self.selected_param_idx < len(params) else None
        
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
            self.request_reprocessing()

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
            self.request_reprocessing()

    def execute_action(self):
        """Executes the primary action (ENTER) based on the current focus."""
        if self.confirming_delete:
            cols = self.get_column_data(1)
            band = cols[self.selected_band_idx]
            if isinstance(band, Band):
                try:
                    # The modules_list used for selection might include [+] buttons
                    # but band.modules is pure AudioModules.
                    # We need to make sure we pop the correct one.
                    # Depth 2 data: Band modules + [+] button. 
                    # selected_module_idx should match band.modules if it's an AudioModule.
                    removed = band.modules.pop(self.selected_module_idx)
                    self.status_msg = f"Deleted {removed.name}."
                    self.selected_module_idx = max(0, self.selected_module_idx - 1)
                    self.selected_param_idx = 0
                    if self.is_playing: self.request_reprocessing()
                except Exception as e:
                    self.status_msg = f"Delete Error: {e}"
            self.confirming_delete = False
            return

        if self.mode == "FILE_PICKER":
            if self.file_list:
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
                            if self.is_playing: self.request_reprocessing()
                        except Exception as e:
                            self.status_msg = f"Load Failed: {e}"
                    else:
                        self.load_audio(item["name"])
            return

        if self.focus_zone == FocusZone.MENU:
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
            self.population.generate_next_generation(self.selected_mix_idx, 0.5, 0.5)
            if self.is_playing: self.request_reprocessing()
            return

        if self.focus_zone == FocusZone.BANDS:
            # Handle confirming FX selection
            if self.selecting_new_fx:
                idx = self.selected_module_idx
                if 0 <= idx < len(self.available_fx_pool):
                    cols = self.get_column_data(1)
                    band = cols[self.selected_band_idx]
                    if isinstance(band, Band):
                        selected_cls = self.available_fx_pool[idx]
                        new_mod = selected_cls()
                        band.modules.append(new_mod)
                        self.status_msg = f"Added {new_mod.name} to {band.name}"
                        if self.is_playing: self.request_reprocessing()
                
                self.selecting_new_fx = False
                self.selected_module_idx = 0
                return

            if self.editing_crossover:
                p = self.get_selected_param()
                if p:
                    self.editing = True
                    self.edit_buffer = str(round(p.current_value, 2))
                return

            modules_list = self.get_column_data(2)
            if 0 <= self.selected_module_idx < len(modules_list):
                item = modules_list[self.selected_module_idx]
                
                # Enter selection mode when [+] Add FX button is clicked
                if item == "[+] Add FX":
                    cols = self.get_column_data(1)
                    band = cols[self.selected_band_idx]
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
                elif self.input_mode == "EXPORT_TRACK":
                    try:
                        self.status_msg = "Exporting..."
                        mix_idx = self.selected_mix_idx % len(self.population.mixes)
                        mix = self.population.mixes[mix_idx]
                        exported_audio = self.engine.process(self.audio_data, mix)
                        exported_audio = np.clip(exported_audio, -1.0, 1.0)
                        filename = f"{self.edit_buffer}.wav"
                        output_path = os.path.join(self.output_dir, filename)
                        sf.write(output_path, exported_audio.T, self.sample_rate)
                        self.status_msg = f"Exported to {filename}"
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
            zones = [FocusZone.MENU, FocusZone.POPULATION, FocusZone.BANDS, FocusZone.PLAYBACK]
            idx = zones.index(self.focus_zone)
            self.focus_zone = zones[(idx + 1) % len(zones)]
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
        if key == KEY_D and self.focus_zone == FocusZone.BANDS and not self.editing:
            modules_list = self.get_column_data(2)
            if 0 <= self.selected_module_idx < len(modules_list):
                item = modules_list[self.selected_module_idx]
                if not isinstance(item, str) and not isinstance(item, Parameter):
                    # It's an AudioModule
                    self.confirming_delete = True
                    self.status_msg = f"Delete {item.name}? (ENTER: Yes, ESC: No)"
                    return

        # Mix Sorting logic (Depth 0 = FocusZone.POPULATION)
        if self.focus_zone == FocusZone.POPULATION:
            if key == KEY_SHIFT_UP:
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
            if key == KEY_UP:
                self.selected_file_idx = (self.selected_file_idx - 1) % max(1, len(self.file_list))
            elif key == KEY_DOWN:
                self.selected_file_idx = (self.selected_file_idx + 1) % max(1, len(self.file_list))
            elif key == KEY_BACKSPACE:
                self.current_path = os.path.abspath(os.path.join(self.current_path, ".."))
                self.refresh_file_list()
            elif key == KEY_ESC:
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
            self.selected_band_idx = (self.selected_band_idx - 1) % len(self.get_column_data(1))
            self.selected_module_idx = 0; self.selected_param_idx = 0; self.editing_crossover = False
            self.selecting_new_fx = False
        elif key == KEY_BRACKET_RIGHT:
            self.selected_band_idx = (self.selected_band_idx + 1) % len(self.get_column_data(1))
            self.selected_module_idx = 0; self.selected_param_idx = 0; self.editing_crossover = False
            self.selecting_new_fx = False
        elif key == KEY_M:
            cols = self.get_column_data(1)
            band = cols[self.selected_band_idx]
            if isinstance(band, Band):
                band.is_muted = not band.is_muted
                self.status_msg = f"Band {band.name} {'Muted' if band.is_muted else 'Unmuted'}"
                if self.is_playing: self.request_reprocessing()
        elif key == KEY_S:
            cols = self.get_column_data(1)
            band = cols[self.selected_band_idx]
            if isinstance(band, Band):
                band.is_soloed = not band.is_soloed
                self.status_msg = f"Band {band.name} {'Soloed' if band.is_soloed else 'Unsoloed'}"
                if self.is_playing: self.request_reprocessing()
        elif key == KEY_K:
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
            if self.sample_rate and self.processed_audio is not None:
                amount = 1 if key == KEY_CTRL_RIGHT else 5
                # Scrub forward safely
                self.play_idx = min(self.processed_audio.shape[1] - 1, self.play_idx + (amount * self.sample_rate))

    def _handle_population_input(self, key: str):
        if key == KEY_LEFT:
            self.selected_mix_idx = (self.selected_mix_idx - 1) % len(self.population.mixes)
            if self.is_playing: self.request_reprocessing()
        elif key == KEY_RIGHT:
            self.selected_mix_idx = (self.selected_mix_idx + 1) % len(self.population.mixes)
            if self.is_playing: self.request_reprocessing()

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
        meta_str = f"File: {meta.filename or 'None'} | {meta.sample_rate}Hz | {meta.channels}ch | {bpm_str}"
        footer_text = f"{self.status_msg}  |  {meta_str}"
        status_panel = Panel(Text(footer_text, style="bold yellow" if self.editing else "italic green"), box=box.SIMPLE)

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
                style = "bold white on blue" if is_sel else "white" if i == self.selected_mix_idx else "dim"
                mix_texts.append(Text(f"{prefix}Mix {i}", style=style))
                
            pop_panel = Panel(Columns(mix_texts, expand=True, align="center"), border_style="magenta" if (self.focus_zone == FocusZone.POPULATION) else "dim")

            # 2. Bands Row
            idx = self.selected_mix_idx % len(self.population.mixes)
            mix = self.population.mixes[idx]
            
            columns = self.get_column_data(1)
            band_layout = Layout(name="bands")
            band_layout.split_row(*[Layout(name=f"col_{i}") for i in range(len(columns))])
            
            for i, item in enumerate(columns):
                is_col_focused = (self.focus_zone == FocusZone.BANDS and self.selected_band_idx == i)
                
                # Rendering a Band (PRE, POST, or Frequency Band)
                band = item
                items = []
                
                # Mute / Solo indicators
                ms_text = Text("")
                if band.is_soloed: ms_text.append("[S]", style="bold black on yellow")
                if band.is_muted: ms_text.append("[M]", style="bold white on red")
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
            elapsed = self.play_idx / self.sample_rate if self.sample_rate else 0
            pct = (elapsed / self.playback_duration) if self.playback_duration > 0 else 0
            bars_filled = int(pct * 20)
            bars_empty = 20 - bars_filled
            progress_bar = f"[{'█' * bars_filled}{'░' * bars_empty}]"
            
            if self.is_playing:
                prog_str = f"{progress_bar} {elapsed:.1f}s / {self.playback_duration:.1f}s"
                audio_panel = Panel(Text(prog_str, style="bold cyan"), title="Audio", border_style="green" if self.focus_zone == FocusZone.PLAYBACK else "dim")
            else:
                prog_str = f"{progress_bar} {elapsed:.1f}s / {self.playback_duration:.1f}s (Stopped)"
                audio_panel = Panel(Text(prog_str, style="dim"), title="Audio", border_style="cyan" if self.focus_zone == FocusZone.PLAYBACK else "dim")

            # Final Assembly of Evolution mode
            layout.split_column(
                Layout(header_panel, size=3),
                Layout(pop_panel, size=3),
                band_layout, # takes remaining ratio=1
                Layout(audio_panel, size=3),
                Layout(status_panel, size=3)
            )
            return layout

def main():
    parser = argparse.ArgumentParser(description="GAIA TUI")
    parser.add_argument("input_file", type=str, nargs="?", help="Optional audio file to load on start")
    args = parser.parse_args()

    audio_data, sr = None, 44100
    if args.input_file:
        try:
            audio_data, sr = sf.read(args.input_file)
            audio_data = audio_data.T if audio_data.ndim > 1 else audio_data.reshape(1, -1)
        except Exception as e:
            print(f"Error loading {args.input_file}: {e}")
            sys.exit(1)

    # Initialize a dummy population
    initial_mixes = [Mix(crossovers=[100.0, 500.0, 2500.0, 8000.0]) for _ in range(5)]
    for m in initial_mixes:
        m.bands[0].modules.append(CompressorModule())
        m.bands[1].modules.append(SaturationModule())
        m.bands[2].modules.append(ExpanderModule())
        m.bands[3].modules.append(CompressorModule())
        m.bands[4].modules.append(TransientShaperModule())

    pop = Population(initial_mixes)
    tui = GaiaTUI(pop, audio_data, sr, ".")

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
