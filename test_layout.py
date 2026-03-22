import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from rich.console import Console
from cli import GaiaTUI, Population
from core.mix import Mix
from core.audio_module import CompressorModule, ExpanderModule, TransientShaperModule, SaturationModule

# Initialize a dummy population
initial_mixes = [Mix(crossovers=[100.0, 500.0, 2500.0, 8000.0]) for _ in range(5)]
for m in initial_mixes:
    m.bands[0].modules.append(CompressorModule())
    m.bands[1].modules.append(SaturationModule())
    m.bands[2].modules.append(ExpanderModule())
    m.bands[3].modules.append(CompressorModule())
    m.bands[4].modules.append(TransientShaperModule())

pop = Population(initial_mixes)
tui = GaiaTUI(pop, None, 44100, ".")
tui.mode = "EVOLUTION"

console = Console()
console.print(tui.render())
