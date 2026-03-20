import sys
import os
import time
from typing import List, Any

# Ensure we can import from core and ga
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console, Group
from rich.text import Text
from rich import box

from core.mix import Mix
from ga.population import Population
from core.audio_module import CompressorModule, ExpanderModule, TransientShaperModule

# --- KEYBOARD LISTENER (Unix/Linux) ---
import tty
import termios

def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        if ch == '\x1b': # Escape sequence
            ch += sys.stdin.read(2)
    except Exception:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Key Constants
KEY_UP = "\x1b[A"
KEY_DOWN = "\x1b[B"
KEY_RIGHT = "\x1b[C"
KEY_LEFT = "\x1b[D"
KEY_ENTER = "\r"
KEY_CTRL_C = "\x03"

class GaiaTUI:
    def __init__(self, population: Population):
        self.population = population
        self.depth = 0 # Current active column
        self.indices = [0, 0, 0, 0] # Cursor position for each column
        self.running = True
        self.status_msg = "Arrows to navigate | ENTER: Evolve (Col 0) or Toggle Lock (Col 2/3)"

    def get_column_data(self, depth: int) -> List[Any]:
        """Returns the list of objects for a specific Miller Column."""
        try:
            if depth == 0:
                return self.population.mixes
            
            # Column 1 depends on Selected Mix (Col 0)
            mix = self.population.mixes[self.indices[0]]
            if depth == 1:
                return ["Crossovers"] + mix.bands
            
            # Column 2 depends on Selected Mix (Col 0) and Selected Branch (Col 1)
            tree_items = ["Crossovers"] + mix.bands
            branch = tree_items[self.indices[1]]
            
            if depth == 2:
                if branch == "Crossovers":
                    return mix.crossover_params
                else: # It's a Band
                    return branch.modules
                
            # Column 3 depends on Selected Module (Col 2)
            if depth == 3:
                if branch == "Crossovers":
                    return [] # No 4th level for crossovers
                modules = branch.modules
                if not modules: return []
                module = modules[self.indices[2]]
                return list(module.parameters.values())
                
        except (IndexError, AttributeError):
            return []
        return []

    def navigate(self, key: str):
        # 1. Update index for current column
        items = self.get_column_data(self.depth)
        
        if key == KEY_UP:
            if items:
                self.indices[self.depth] = (self.indices[self.depth] - 1) % len(items)
        elif key == KEY_DOWN:
            if items:
                self.indices[self.depth] = (self.indices[self.depth] + 1) % len(items)
        elif key == KEY_LEFT:
            if self.depth > 0:
                self.depth -= 1
        elif key == KEY_RIGHT:
            # Check if next level has data
            if self.depth < 3:
                next_items = self.get_column_data(self.depth + 1)
                if next_items:
                    self.depth += 1
                    # Ensure child index is valid
                    if self.indices[self.depth] >= len(next_items):
                        self.indices[self.depth] = 0
        elif key == KEY_ENTER:
            self.execute_action()
        elif key == KEY_CTRL_C:
            self.running = False

    def execute_action(self):
        """Action logic for Enter key."""
        if self.depth == 0:
            # Evolve whole population from this parent
            self.population.generate_next_generation(self.indices[0], 0.5, 0.5)
            self.status_msg = f"Evolved generation {self.population.generation_count} from Mix {self.indices[0]}!"
        
        elif self.depth == 2:
            # Toggle Lock on Crossover
            items = self.get_column_data(2)
            if items and self.indices[1] == 0: # Crossovers branch
                param = items[self.indices[2]]
                param.is_locked = not param.is_locked
                self.status_msg = f"Toggled Lock: {param.name} is now {'LOCKED' if param.is_locked else 'FREE'}"
        
        elif self.depth == 3:
            # Toggle Lock on Module Parameter
            items = self.get_column_data(3)
            if items:
                param = items[self.indices[3]]
                param.is_locked = not param.is_locked
                self.status_msg = f"Toggled Lock: {param.name} is now {'LOCKED' if param.is_locked else 'FREE'}"

    def render_panel(self, depth: int, title: str) -> Panel:
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
                    label = f"{item.name}: {item.current_value:.1f} Hz"
                else: # Modules
                    label = f"[{i}] {item.name}"
            elif depth == 3: # Module Params
                label = f"{item.name}: {item.current_value:7.2f}"

            # 2. Status Indicators
            locked = " 🔒" if getattr(item, 'is_locked', False) else ""
            
            # 3. Styling
            text_style = "white"
            if i == self.indices[depth]:
                if is_active_col:
                    text_style = "bold white on blue"
                    label = f"▶ {label}"
                else:
                    text_style = "bold yellow" # Path highlight
                    label = f"  {label}"
            else:
                label = f"  {label}"

            content.append(Text(f"{label}{locked}", style=text_style))

        if not items:
            content.append(Text("  (None)", style="dim"))

        border_style = "cyan" if is_active_col else "white"
        return Panel(Group(*content), title=f"[bold]{title}[/bold]", border_style=border_style, padding=(1, 1))

def main():
    # Setup initial population
    initial_mixes = [Mix(crossovers=[100.0, 500.0, 2500.0]) for _ in range(5)]
    for m in initial_mixes:
        m.bands[0].modules.append(CompressorModule())
        m.bands[1].modules.append(TransientShaperModule())
        m.bands[2].modules.append(ExpanderModule())

    pop = Population(initial_mixes)
    tui = GaiaTUI(pop)
    
    # Setup Layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    layout["body"].split_row(
        Layout(name="c0"), Layout(name="c1"), Layout(name="c2"), Layout(name="c3")
    )

    with Live(layout, refresh_per_second=20, screen=True):
        while tui.running:
            # Update Header
            layout["header"].update(Panel(Text(f"GAIA EVOLUTION SYSTEM | Gen {pop.generation_count}", justify="center", style="bold magenta"), box=box.SIMPLE))
            
            # Update Body
            layout["c0"].update(tui.render_panel(0, "Population"))
            layout["c1"].update(tui.render_panel(1, "Mix Tree"))
            layout["c2"].update(tui.render_panel(2, "Modules/XO"))
            layout["c3"].update(tui.render_panel(3, "Params"))
            
            # Update Footer
            layout["footer"].update(Panel(Text(tui.status_msg, style="italic green"), box=box.SIMPLE))
            
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
