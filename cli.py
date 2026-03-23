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
from ga.population import Population
from core.audio_module import CompressorModule, ExpanderModule, TransientShaperModule, SaturationModule
from audio.engine import Engine

# Constants
KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_RIGHT = "RIGHT"
KEY_LEFT = "LEFT"
KEY_ENTER = "ENTER"
KEY_TAB = "TAB"
KEY_CTRL_C = "CTRL_C"
KEY_ESC = "ESC"
KEY_SPACE = " "
KEY_BACKSPACE = "BACKSPACE"
KEY_L = "L"
KEY_E = "E"
KEY_P = "P"
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

    if b == b'\x1b[A': return KEY_UP
    if b == b'\x1b[B': return KEY_DOWN
    if b == b'\x1b[C': return KEY_RIGHT
    if b == b'\x1b[D': return KEY_LEFT
    if b == b'\x1b': return KEY_ESC
    if b in (b'\r', b'\n'): return KEY_ENTER
    if b == b'\t': return KEY_TAB
    if b == b'\x03': return KEY_CTRL_C
    if b in (b'\x08', b'\x7f'): return KEY_BACKSPACE
    if b == b' ': return KEY_SPACE
    
    try:
        decoded = b.decode('ascii').upper()
        if decoded == '[': return KEY_BRACKET_LEFT
        if decoded == ']': return KEY_BRACKET_RIGHT
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
            MenuItem(title="File", action=lambda: None),
            MenuItem(title="Analysis", action=lambda: None),
            MenuItem(title="Settings", action=lambda: None),
        ])
        self.metadata = TrackMetadata()
        self.focus_row = 0 # 0: Population, 1: Bands
        self.selected_mix_idx = 0
        self.selected_file_idx = 0 # Separate index for file browser
        self.selected_band_idx = 0
        self.selected_module_idx = 0
        self.selected_param_idx = 0
        self.editing_crossover = False
        
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
        self.status_msg = "Arrows: Navigate/Adjust | TAB: Row | [ ]: Band | ENTER: Edit | SPACE: Play/Stop"

    def refresh_file_list(self):
        """Scans path for audio files."""
        exts = (".wav", ".flac", ".mp3", ".aiff", ".ogg")
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
                
                self.stream = sd.OutputStream(
                    samplerate=self.sample_rate, 
                    channels=self.processed_audio.shape[0], 
                    callback=self.audio_callback
                )
                self.stream.start()
                self.is_playing = True
                self.status_msg = "Playing..."
            except Exception as e:
                self.status_msg = f"Error: {e}"

    def get_selected_param(self):
        if self.mode != "EVOLUTION": return None
        idx = self.selected_mix_idx % len(self.population.mixes)
        mix = self.population.mixes[idx]
        if self.editing_crossover:
            return mix.crossover_params[self.selected_band_idx] if self.selected_band_idx < len(mix.crossover_params) else None
        band = mix.bands[self.selected_band_idx]
        if band.modules and self.selected_module_idx < len(band.modules):
            mod = band.modules[self.selected_module_idx]
            params = list(mod.parameters.values())
            return params[self.selected_param_idx] if self.selected_param_idx < len(params) else None
        return None

    def adjust_value(self, direction: int):
        p = self.get_selected_param()
        if p:
            step = (p.max_bound - p.min_bound) * 0.05 * direction
            p.current_value = max(p.min_bound, min(p.max_bound, p.current_value + step))
            if self.editing_crossover: 
                idx = self.selected_mix_idx % len(self.population.mixes)
                self.population.mixes[idx].sort_crossovers()
            if self.is_playing:
                self.request_reprocessing()

    def set_value(self, val: float):
        p = self.get_selected_param()
        if p:
            p.current_value = max(p.min_bound, min(p.max_bound, val))
            if self.editing_crossover:
                idx = self.selected_mix_idx % len(self.population.mixes)
                self.population.mixes[idx].sort_crossovers()
            if self.is_playing:
                self.request_reprocessing()

    def navigate(self, key: str):
        if self.editing:
            if key == KEY_ENTER:
                try: self.set_value(float(self.edit_buffer))
                except: pass
                self.editing = False; self.edit_buffer = ""
            elif key == KEY_ESC:
                self.editing = False; self.edit_buffer = ""
            elif key == KEY_BACKSPACE:
                self.edit_buffer = self.edit_buffer[:-1]
            elif key and len(key) == 1 and key != " ":
                self.edit_buffer += key
            return

        if key == KEY_TAB and self.mode == "EVOLUTION":
            if self.focus_zone == FocusZone.BANDS:
                num_bands = len(self.population.mixes[0].bands)
                if self.selected_band_idx < num_bands - 1:
                    self.selected_band_idx += 1
                else:
                    self.focus_zone = FocusZone.PLAYBACK
                    self.selected_band_idx = 0
            else:
                zones = list(FocusZone)
                idx = zones.index(self.focus_zone)
                next_zone = zones[(idx + 1) % len(zones)]
                self.focus_zone = next_zone
                if next_zone == FocusZone.BANDS:
                    self.selected_band_idx = 0
            return

        if key == KEY_SPACE:
            self.toggle_playback()
            return

        # If in FILE_PICKER, handle its inputs directly
        if self.mode == "FILE_PICKER":
            if key == KEY_UP:
                self.selected_file_idx = (self.selected_file_idx - 1) % max(1, len(self.file_list))
            elif key == KEY_DOWN:
                self.selected_file_idx = (self.selected_file_idx + 1) % max(1, len(self.file_list))
            elif key == KEY_ENTER:
                if self.file_list:
                    item = self.file_list[self.selected_file_idx]
                    if item["type"] == "dir":
                        self.current_path = os.path.abspath(os.path.join(self.current_path, item["name"]))
                        self.refresh_file_list()
                    else:
                        self.load_audio(item["name"])
            elif key == KEY_BACKSPACE:
                self.current_path = os.path.abspath(os.path.join(self.current_path, ".."))
                self.refresh_file_list()
            return

        # Evolution Global Shortcuts
        if key == KEY_L:
            self.mode = "FILE_PICKER"
            self.refresh_file_list()
            return
        elif key == KEY_BRACKET_LEFT:
            self.selected_band_idx = (self.selected_band_idx - 1) % len(self.population.mixes[0].bands)
            self.selected_module_idx = 0; self.selected_param_idx = 0; self.editing_crossover = False
        elif key == KEY_BRACKET_RIGHT:
            self.selected_band_idx = (self.selected_band_idx + 1) % len(self.population.mixes[0].bands)
            self.selected_module_idx = 0; self.selected_param_idx = 0; self.editing_crossover = False

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
        if key == KEY_LEFT:
            self.header_menu.selected_index = (self.header_menu.selected_index - 1) % len(self.header_menu.items)
        elif key == KEY_RIGHT:
            self.header_menu.selected_index = (self.header_menu.selected_index + 1) % len(self.header_menu.items)
        elif key == KEY_ENTER:
            item = self.header_menu.items[self.header_menu.selected_index]
            if item.action:
                item.action()

    def _handle_playback_input(self, key: str):
        if key == KEY_LEFT:
            if self.sample_rate:
                # Scrub 5s back
                self.play_idx = max(0, self.play_idx - (5 * self.sample_rate))
        elif key == KEY_RIGHT:
            if self.sample_rate and self.processed_audio is not None:
                # Scrub 5s forward
                self.play_idx = min(self.processed_audio.shape[1] - 1, self.play_idx + (5 * self.sample_rate))

    def _handle_population_input(self, key: str):
        if key == KEY_LEFT:
            self.selected_mix_idx = (self.selected_mix_idx - 1) % len(self.population.mixes)
            if self.is_playing: self.request_reprocessing()
        elif key == KEY_RIGHT:
            self.selected_mix_idx = (self.selected_mix_idx + 1) % len(self.population.mixes)
            if self.is_playing: self.request_reprocessing()
        elif key == KEY_ENTER:
            self.population.generate_next_generation(self.selected_mix_idx, 0.5, 0.5)
            if self.is_playing: self.request_reprocessing()

    def _handle_bands_input(self, key: str):
        mix = self.population.mixes[self.selected_mix_idx % len(self.population.mixes)]
        band = mix.bands[self.selected_band_idx]
        if key == KEY_UP:
            if self.selected_param_idx > 0:
                self.selected_param_idx -= 1
            elif self.selected_module_idx > 0:
                self.selected_module_idx -= 1
                self.selected_param_idx = len(band.modules[self.selected_module_idx].parameters) - 1
            elif not self.editing_crossover:
                self.editing_crossover = True
        elif key == KEY_DOWN:
            if self.editing_crossover:
                self.editing_crossover = False
                self.selected_module_idx = 0
                self.selected_param_idx = 0
            elif band.modules:
                mod = band.modules[self.selected_module_idx]
                if self.selected_param_idx < len(mod.parameters) - 1:
                    self.selected_param_idx += 1
                elif self.selected_module_idx < len(band.modules) - 1:
                    self.selected_module_idx += 1
                    self.selected_param_idx = 0
        elif key == KEY_LEFT:
            self.adjust_value(-1)
        elif key == KEY_RIGHT:
            self.adjust_value(1)
        elif key == KEY_ENTER:
            p = self.get_selected_param()
            if p:
                self.editing = True
                self.edit_buffer = str(round(p.current_value, 2))

    def render(self):
        # Build the UI Layout dynamically to avoid out-of-bounds rendering
        layout = Layout()
        
        # Common Header and Status
        menu_texts = []
        for i, item in enumerate(self.header_menu.items):
            style = "bold white on blue" if (self.focus_zone == FocusZone.MENU and i == self.header_menu.selected_index) else "white"
            menu_texts.append(Text(f" [{item.title}] ", style=style))
            
        header_text = Text(f"GAIA GENMIX | Gen {self.population.generation_count}", style="bold magenta", justify="center")
        header_content = Columns([header_text] + menu_texts, expand=True, align="center")
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
                
            pop_panel = Panel(Columns(mix_texts, expand=True, align="center"), title="Population", border_style="magenta" if (self.focus_zone == FocusZone.POPULATION) else "dim")

            # 2. Bands Row (using a horizontal layout to prevent wrapping)
            idx = self.selected_mix_idx % len(self.population.mixes)
            mix = self.population.mixes[idx]
            band_layout = Layout(name="bands")
            band_layout.split_row(*[Layout(name=f"band_{i}") for i in range(len(mix.bands))])
            
            for i, band in enumerate(mix.bands):
                is_band_sel = (self.focus_zone == FocusZone.BANDS and self.selected_band_idx == i)
                items = []
                
                # Crossover rendering
                if i < len(mix.crossover_params):
                    xo = mix.crossover_params[i]
                    is_xo_sel = is_band_sel and self.editing_crossover
                    txt = self.edit_buffer if (is_xo_sel and self.editing) else f"{xo.current_value:.0f}"
                    style = "bold black on yellow" if (is_xo_sel and self.editing) else "bold white on blue" if is_xo_sel else "cyan"
                    items.append(Text(f"XO: {txt}Hz", style=style, overflow="ellipsis", no_wrap=True))
                    items.append(Text("-" * 10, style="dim", no_wrap=True))
                else:
                    items.append(Text("XO: Nyq", style="dim", overflow="ellipsis", no_wrap=True))
                    items.append(Text("-" * 10, style="dim", no_wrap=True))

                # Modules rendering
                for m_idx, mod in enumerate(band.modules):
                    is_mod_sel = is_band_sel and not self.editing_crossover and self.selected_module_idx == m_idx
                    mod_name = mod.name[:7] + "." if len(mod.name) > 8 else mod.name
                    style = "bold white on blue" if is_mod_sel else "white"
                    items.append(Text(f"[{mod_name}]", style=style, overflow="ellipsis", no_wrap=True))
                    
                    for p_idx, (p_name, p) in enumerate(mod.parameters.items()):
                        is_p_sel = is_mod_sel and self.selected_param_idx == p_idx
                        val_txt = self.edit_buffer if (is_p_sel and self.editing) else f"{p.current_value:.1f}"
                        short_name = p_name[:5]
                        style = "bold black on yellow" if (is_p_sel and self.editing) else "bold white on cyan" if is_p_sel else "dim"
                        items.append(Text(f" {short_name}: {val_txt}", style=style, overflow="ellipsis", no_wrap=True))
                        
                band_panel = Panel(Group(*items), title=f"B{i+1}", border_style="blue" if (self.focus_zone == FocusZone.BANDS and is_band_sel) else "dim")
                band_layout[f"band_{i}"].update(band_panel)

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
    parser = argparse.ArgumentParser(description="Gaia GenMix TUI")
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
