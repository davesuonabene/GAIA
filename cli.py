import sys
import os
import argparse
import numpy as np
import soundfile as sf
import sounddevice as sd
from typing import List, Any

# Ensure we can import from core and ga
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rich.live import Live
from rich.panel import Panel
from rich.console import Group
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich import box

from core.mix import Mix
from ga.population import Population
from core.audio_module import CompressorModule, ExpanderModule, TransientShaperModule
from audio.engine import Engine

# Key Constants
KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_RIGHT = "RIGHT"
KEY_LEFT = "LEFT"
KEY_ENTER = "ENTER"
KEY_CTRL_C = "CTRL_C"
KEY_ESC = "ESC"
KEY_PLUS = "+"
KEY_EQUAL = "="
KEY_MINUS = "-"
KEY_SPACE = " "
KEY_BACKSPACE = "BACKSPACE"
KEY_E = "E"
KEY_P = "P"
KEY_L = "L"

# --- KEYBOARD LISTENER (Cross-Platform) ---
try:
    import msvcrt
    def get_key():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b'\x00', b'\xe0'): # Arrow key prefix
                ch2 = msvcrt.getch()
                if ch2 == b'H': return KEY_UP
                elif ch2 == b'P': return KEY_DOWN
                elif ch2 == b'M': return KEY_RIGHT
                elif ch2 == b'K': return KEY_LEFT
            elif ch == b'\r': return KEY_ENTER
            elif ch == b'\x03': return KEY_CTRL_C
            elif ch == b'\x1b': return KEY_ESC
            elif ch == b'\x08': return KEY_BACKSPACE
            elif ch in (b'+', b'='): return KEY_PLUS
            elif ch == b'-': return KEY_MINUS
            elif ch == b' ': return KEY_SPACE
            elif ch in (b'e', b'E'): return KEY_E
            elif ch in (b'p', b'P'): return KEY_P
            elif ch in (b'l', b'L'): return KEY_L
            else:
                try: return ch.decode('ascii')
                except: return None
        return None
except ImportError:
    import tty
    import termios
    import select
    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            # Use os.read to avoid buffering issues with sys.stdin
            ch_bytes = os.read(fd, 1)
            if not ch_bytes:
                return None
            ch = ch_bytes.decode('ascii', errors='ignore')
            
            if ch == '\x1b': # Escape sequence
                # Wait for the next byte to distinguish ESC key from Arrow keys
                r, _, _ = select.select([fd], [], [], 0.1)
                if r:
                    ch2 = os.read(fd, 1).decode('ascii', errors='ignore')
                    if ch2 in ('[', 'O'):
                        r2, _, _ = select.select([fd], [], [], 0.1)
                        if r2:
                            ch3 = os.read(fd, 1).decode('ascii', errors='ignore')
                            if ch3 == 'A': return KEY_UP
                            elif ch3 == 'B': return KEY_DOWN
                            elif ch3 == 'C': return KEY_RIGHT
                            elif ch3 == 'D': return KEY_LEFT
                return KEY_ESC
            elif ch == '\r': return KEY_ENTER
            elif ch == '\x03': return KEY_CTRL_C
            elif ch in ('\x08', '\x7f'): return KEY_BACKSPACE
            elif ch == '=': return KEY_PLUS
            elif ch == '-': return KEY_MINUS
            elif ch == ' ': return KEY_SPACE
            elif ch in ('e', 'E'): return KEY_E
            elif ch in ('p', 'P'): return KEY_P
            elif ch in ('l', 'L'): return KEY_L
            else:
                return ch
        except Exception:
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

class GaiaTUI:
    def __init__(self, population: Population, audio_data: np.ndarray, sample_rate: int, output_dir: str):
        self.population = population
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.depth = 0 # Current active column (0 to 3)
        self.indices = [0, 0, 0, 0] # Cursor position for each column
        self.running = True
        self.engine = Engine(sample_rate=self.sample_rate) if self.sample_rate else None
        self.is_playing = False
        
        # Mode and File Browser
        self.mode = "EVOLUTION" if self.audio_data is not None else "FILE_PICKER"
        self.current_path = os.path.abspath(".")
        self.file_list = []
        if self.mode == "FILE_PICKER":
            self.refresh_file_list()
        
        # Inline Editing State
        self.editing = False
        self.edit_buffer = ""
        
        self.status_msg_default = "Arrows: Navigate | SPACE: Lock | ENTER: Edit/Evolve | +/-: Adjust | E: Export | P: Play/Stop | L: Browser"
        self.status_msg = self.status_msg_default
        if self.mode == "FILE_PICKER":
            self.status_msg = "Arrows: Select | ENTER: Open/Load | P: Preview | L: Refresh | Backspace: Up"

    def refresh_file_list(self):
        """Scans self.current_path for audio files and directories."""
        extensions = (".wav", ".flac", ".mp3", ".aiff", ".ogg")
        try:
            items = os.listdir(self.current_path)
            # Filter and sort: Directories first, then supported audio files
            dirs = sorted([f for f in items if os.path.isdir(os.path.join(self.current_path, f)) and not f.startswith(".")])
            files = sorted([f for f in items if f.lower().endswith(extensions)])
            
            self.file_list = [{"name": "..", "type": "dir"}] if os.path.dirname(self.current_path) != self.current_path else []
            self.file_list += [{"name": d, "type": "dir"} for d in dirs]
            self.file_list += [{"name": f, "type": "file"} for f in files]
            
            self.indices = [0, 0, 0, 0]
            self.depth = 0
        except Exception as e:
            self.status_msg = f"Error accessing {self.current_path}: {e}"
            self.file_list = [{"name": "..", "type": "dir"}]

    def load_audio(self, filename: str):
        """Loads a new audio file and switches to EVOLUTION mode."""
        if self.is_playing:
            sd.stop()
            self.is_playing = False
            
        full_path = os.path.join(self.current_path, filename)
        try:
            self.status_msg = f"Loading {filename}..."
            audio_data, sample_rate = sf.read(full_path)
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            else:
                audio_data = audio_data.T
            self.audio_data = audio_data
            self.sample_rate = sample_rate
            self.engine = Engine(sample_rate=self.sample_rate)
            self.mode = "EVOLUTION"
            self.indices = [0, 0, 0, 0]
            self.depth = 0
            self.status_msg = f"Loaded {filename}! " + self.status_msg_default
        except Exception as e:
            self.status_msg = f"Error loading {filename}: {e}"

    def process_current_mix(self) -> np.ndarray:
        if self.audio_data is None:
            raise ValueError("No audio data loaded.")
        mix = self.population.mixes[self.indices[0]]
        return self.engine.process(self.audio_data, mix)

    def _reset_child_indices(self):
        """Resets the indices for all columns deeper than the current active column."""
        for i in range(self.depth + 1, 4):
            self.indices[i] = 0

    def get_column_data(self, depth: int) -> List[Any]:
        """Returns the list of objects for a specific Miller Column."""
        try:
            if depth == 0:
                return self.population.mixes
            
            mix = self.population.mixes[self.indices[0]]
            if depth == 1:
                return ["Crossovers"] + mix.bands
            
            tree_items = ["Crossovers"] + mix.bands
            branch = tree_items[self.indices[1]]
            
            if depth == 2:
                if branch == "Crossovers":
                    return mix.crossover_params
                else: 
                    return branch.modules
                
            if depth == 3:
                if branch == "Crossovers":
                    return [] 
                modules = branch.modules
                if not modules: return []
                module = modules[self.indices[2]]
                return list(module.parameters.values())
                
        except (IndexError, AttributeError):
            return []
        return []

    def navigate(self, key: str):
        if not key:
            return
            
        # File Browser specific navigation
        if self.mode == "FILE_PICKER":
            if key == KEY_UP:
                if self.file_list:
                    self.indices[0] = (self.indices[0] - 1) % len(self.file_list)
                    if self.is_playing: # Stop preview if moving
                        sd.stop()
                        self.is_playing = False
            elif key == KEY_DOWN:
                if self.file_list:
                    self.indices[0] = (self.indices[0] + 1) % len(self.file_list)
                    if self.is_playing: # Stop preview if moving
                        sd.stop()
                        self.is_playing = False
            elif key == KEY_ENTER:
                if self.file_list:
                    item = self.file_list[self.indices[0]]
                    if item["type"] == "dir":
                        if self.is_playing:
                            sd.stop()
                            self.is_playing = False
                        self.current_path = os.path.abspath(os.path.join(self.current_path, item["name"]))
                        self.refresh_file_list()
                    else:
                        self.load_audio(item["name"])
            elif key == KEY_BACKSPACE:
                if self.is_playing:
                    sd.stop()
                    self.is_playing = False
                self.current_path = os.path.abspath(os.path.join(self.current_path, ".."))
                self.refresh_file_list()
            elif key == KEY_L:
                self.refresh_file_list()
            elif key == KEY_P:
                self.toggle_preview()
            elif key == KEY_CTRL_C:
                if self.is_playing:
                    sd.stop()
                self.running = False
            return

        # Global 'Browser' key to switch to file browser
        if key == KEY_L:
            if self.is_playing:
                sd.stop()
                self.is_playing = False
            self.mode = "FILE_PICKER"
            self.refresh_file_list()
            self.status_msg = "Arrows: Select | ENTER: Open/Load | P: Preview | L: Refresh | Backspace: Up"
            return

        # If we are currently INLINE EDITING a parameter:
        if self.editing:
            if key == KEY_ENTER:
                try:
                    new_val = float(self.edit_buffer)
                    self.set_value_directly(new_val)
                except ValueError:
                    self.status_msg = "Invalid number. Edit cancelled."
                self.editing = False
                self.edit_buffer = ""
            elif key == KEY_ESC:
                self.editing = False
                self.edit_buffer = ""
                self.status_msg = "Edit cancelled."
            elif key == KEY_BACKSPACE:
                self.edit_buffer = self.edit_buffer[:-1]
            elif isinstance(key, str) and len(key) == 1 and key.isprintable():
                self.edit_buffer += key
            return

        # Normal Navigation Logic
        items = self.get_column_data(self.depth)
        
        if key == KEY_UP:
            if len(items) > 0:
                self.indices[self.depth] = (self.indices[self.depth] - 1) % len(items)
                self._reset_child_indices()
        elif key == KEY_DOWN:
            if len(items) > 0:
                self.indices[self.depth] = (self.indices[self.depth] + 1) % len(items)
                self._reset_child_indices()
        elif key == KEY_LEFT:
            if self.depth > 0:
                self.depth -= 1
                self._reset_child_indices()
        elif key == KEY_RIGHT:
            if self.depth < 3:
                next_items = self.get_column_data(self.depth + 1)
                if len(next_items) > 0:
                    self.depth += 1
                    if self.indices[self.depth] >= len(next_items):
                        self.indices[self.depth] = 0
        elif key == KEY_ENTER:
            self.execute_action()
        elif key == KEY_SPACE:
            self.toggle_lock()
        elif key == KEY_PLUS:
            self.adjust_value(1)
        elif key == KEY_MINUS:
            self.adjust_value(-1)
        elif key == KEY_CTRL_C:
            self.running = False
        elif key == KEY_E:
            if self.audio_data is None:
                self.status_msg = "Cannot export: No audio loaded."
                return
            self.status_msg = f"Exporting Mix {self.indices[0]}..."
            try:
                processed_audio = self.process_current_mix()
                os.makedirs(self.output_dir, exist_ok=True)
                filename = os.path.join(self.output_dir, f"gen_{self.population.generation_count}_mix_{self.indices[0]}.wav")
                sf.write(filename, processed_audio.T, self.sample_rate)
                self.status_msg = f"Exported to {filename}!"
            except Exception as e:
                self.status_msg = f"Export failed: {e}"
        elif key == KEY_P:
            if self.audio_data is None:
                self.status_msg = "Cannot play: No audio loaded."
                return
            if self.is_playing:
                sd.stop()
                self.is_playing = False
                self.status_msg = "Playback stopped."
            else:
                self.status_msg = "Processing audio for playback..."
                try:
                    processed_audio = self.process_current_mix()
                    sd.play(processed_audio.T, self.sample_rate)
                    self.is_playing = True
                    self.status_msg = "Playing mix..."
                except Exception as e:
                    self.status_msg = f"Playback failed: {e}"

    def toggle_preview(self):
        """Plays or stops a preview of the selected file in the browser."""
        if not self.file_list: return
        item = self.file_list[self.indices[0]]
        if item["type"] != "file": return

        if self.is_playing:
            sd.stop()
            self.is_playing = False
            self.status_msg = "Preview stopped."
        else:
            full_path = os.path.join(self.current_path, item["name"])
            self.status_msg = f"Previewing {item['name']}..."
            try:
                # Read temporarily for preview
                data, fs = sf.read(full_path)
                sd.play(data, fs)
                self.is_playing = True
            except Exception as e:
                self.status_msg = f"Preview failed: {e}"

    def get_selected_param(self) -> Any:
        """Returns the currently selected Parameter object, or None if not at a parameter level."""
        if self.mode != "EVOLUTION": return None
        if self.depth == 2:
            items = self.get_column_data(2)
            if items and self.indices[1] == 0: # Crossovers branch
                return items[self.indices[2]]
        elif self.depth == 3:
            items = self.get_column_data(3)
            if items:
                return items[self.indices[3]]
        return None

    def toggle_lock(self):
        """Toggles the lock status of the currently selected parameter."""
        param = self.get_selected_param()
        if param:
            param.is_locked = not param.is_locked
            self.status_msg = f"Toggled Lock: {param.name} is now {'LOCKED' if param.is_locked else 'FREE'}"

    def set_value_directly(self, new_val: float):
        """Sets the value of the selected parameter, respecting bounds."""
        param = self.get_selected_param()
        if param:
            param.current_value = max(param.min_bound, min(param.max_bound, new_val))
            self.status_msg = f"Set {param.name} to {param.current_value:.2f}"
            
            # If it's a crossover, resort and rename to maintain logic, and update cursor
            if self.depth == 2 and self.indices[1] == 0:
                mix = self.population.mixes[self.indices[0]]
                mix.sort_crossovers()
                self.indices[2] = mix.crossover_params.index(param)

    def adjust_value(self, direction: int):
        """Manually steps a selected parameter's value up or down."""
        param = self.get_selected_param()
        if not param:
            return
            
        if self.depth == 2: # Crossover
            step = (param.max_bound - param.min_bound) * 0.02 * direction
        else: # Module param
            step = (param.max_bound - param.min_bound) * 0.05 * direction
            
        new_val = param.current_value + step
        param.current_value = max(param.min_bound, min(param.max_bound, new_val))
        self.status_msg = f"Adjusted {param.name} to {param.current_value:.2f}"
        
        if self.depth == 2 and self.indices[1] == 0:
            mix = self.population.mixes[self.indices[0]]
            mix.sort_crossovers()
            self.indices[2] = mix.crossover_params.index(param)

    def execute_action(self):
        """Action logic for Enter key."""
        if self.mode != "EVOLUTION": return
        if self.depth == 0:
            # Evolve whole population from this parent
            self.population.generate_next_generation(self.indices[0], 0.5, 0.5)
            self._reset_child_indices()
            self.status_msg = f"Evolved generation {self.population.generation_count} from Mix {self.indices[0]}!"
        elif self.depth in (2, 3):
            param = self.get_selected_param()
            if param:
                # Start inline editing
                self.editing = True
                self.edit_buffer = str(round(param.current_value, 2))
                self.status_msg = f"Editing {param.name} (Min: {param.min_bound}, Max: {param.max_bound}). ENTER to confirm, ESC to cancel."

    def render_column_content(self, depth: int) -> Group:
        if self.mode == "FILE_PICKER":
            if depth == 0:
                content = []
                # Show current path as a header
                content.append(Text(f" Path: {self.current_path}", style="bold yellow"))
                content.append(Text("-" * 20, style="dim"))
                
                for i, item in enumerate(self.file_list):
                    prefix = "▶ " if i == self.indices[0] else "  "
                    style = "bold white on blue" if i == self.indices[0] else "white"
                    
                    display_name = item["name"]
                    if item["type"] == "dir":
                        display_name = f"[DIR] {display_name}"
                    
                    content.append(Text(f"{prefix}{display_name}", style=style))
                
                if not self.file_list:
                    content.append(Text("  (No audio files or directories found)", style="dim"))
                return Group(*content)
            else:
                return Group(Text("  ---", style="dim"))

        items = self.get_column_data(depth)
        is_active_col = (self.depth == depth)
        
        content = []
        for i, item in enumerate(items):
            # 1. Determine Label
            label = ""
            if depth == 0: label = f"Mix {i}"
            elif depth == 1: label = item if isinstance(item, str) else f"Band: {item.name}"
            elif depth == 2:
                if self.indices[1] == 0: # Crossovers
                    if self.editing and is_active_col and i == self.indices[depth]:
                        label = f"{item.name}: {self.edit_buffer} Hz █"
                    else:
                        label = f"{item.name}: {item.current_value:.1f} Hz"
                else: # Modules
                    label = f"[{i}] {item.name}"
            elif depth == 3: # Module Params
                if self.editing and is_active_col and i == self.indices[depth]:
                    label = f"{item.name}: {self.edit_buffer} █"
                else:
                    label = f"{item.name}: {item.current_value:7.2f}"

            # 2. Status Indicators
            locked = " 🔒" if getattr(item, 'is_locked', False) else ""
            
            # 3. Styling
            text_style = "white"
            prefix = "  "
            if i == self.indices[depth]:
                if is_active_col:
                    if self.editing:
                        text_style = "bold black on yellow"
                    else:
                        text_style = "bold white on blue"
                    prefix = "▶ "
                else:
                    text_style = "bold grey74" # Path highlight
                    prefix = "  "

            content.append(Text(f"{prefix}{label}{locked}", style=text_style))

        if not items:
            content.append(Text("  (None)", style="dim"))

        return Group(*content)

def main():
    parser = argparse.ArgumentParser(description="GenMix Gaia TUI")
    parser.add_argument("input_file", type=str, nargs="?", help="Path to the input audio file")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files")
    args = parser.parse_args()

    audio_data = None
    sample_rate = 44100
    if args.input_file:
        try:
            audio_data, sample_rate = sf.read(args.input_file)
            # soundfile returns (samples, channels). pedalboard needs (channels, samples).
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(1, -1)
            else:
                audio_data = audio_data.T
        except Exception as e:
            print(f"Error loading audio file: {e}")
            sys.exit(1)

    # Setup initial population (4 crossovers = 5 bands)
    initial_mixes = [Mix(crossovers=[100.0, 500.0, 2500.0, 8000.0]) for _ in range(5)]
    for m in initial_mixes:
        m.bands[0].modules.append(CompressorModule())
        m.bands[1].modules.append(TransientShaperModule())
        m.bands[2].modules.append(ExpanderModule())
        m.bands[3].modules.append(CompressorModule())
        m.bands[4].modules.append(TransientShaperModule())

    pop = Population(initial_mixes)
    tui = GaiaTUI(pop, audio_data, sample_rate, args.output_dir)

    with Live(auto_refresh=False, screen=True) as live:
        while tui.running:
            # 1. Header
            header = Panel(
                Align.center(f"[bold magenta]GAIA EVOLUTION SYSTEM | Gen {pop.generation_count}[/bold magenta]"), 
                box=box.DOUBLE
            )
            
            # 2. Miller Columns
            c0 = Panel(tui.render_column_content(0), title="[bold]Population[/bold]", border_style="cyan" if tui.depth == 0 else "dim")
            c1 = Panel(tui.render_column_content(1), title="[bold]Mix Tree[/bold]", border_style="cyan" if tui.depth == 1 else "dim")
            c2 = Panel(tui.render_column_content(2), title="[bold]Modules/XO[/bold]", border_style="cyan" if tui.depth == 2 else "dim")
            c3 = Panel(tui.render_column_content(3), title="[bold]Params[/bold]", border_style="cyan" if tui.depth == 3 else "dim")
            
            columns = Columns([c0, c1, c2, c3], expand=True)
            
            # 3. Footer
            footer_style = "bold yellow" if tui.editing else "italic green"
            footer = Panel(Text(tui.status_msg, style=footer_style), box=box.SIMPLE)
            
            # Assemble & Render
            renderable = Group(header, columns, footer)
            live.update(renderable, refresh=True)
            
            # Input
            key = get_key()
            if key:
                tui.navigate(key)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    print("\nExited Gaia.")