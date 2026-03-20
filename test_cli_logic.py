import sys
import os
from rich.console import Console

# Ensure we can import from core and ga
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.mix import Mix
from ga.population import Population
from core.audio_module import CompressorModule, ExpanderModule, TransientShaperModule

# Mock UIState class (simplified)
class MockUIState:
    def __init__(self, population):
        self.population = population
        self.depth = 0
        self.indices = [0, 0, 0, 0]

    def get_items(self, depth: int):
        try:
            if depth == 0: return self.population.mixes
            if depth == 1:
                mix = self.population.mixes[self.indices[0]]
                return ["Crossovers"] + mix.bands
            if depth == 2:
                mix = self.population.mixes[self.indices[0]]
                d1_items = ["Crossovers"] + mix.bands
                d1_sel = d1_items[self.indices[1]]
                if d1_sel == "Crossovers": return mix.crossover_params
                else: return d1_sel.modules
            if depth == 3:
                mix = self.population.mixes[self.indices[0]]
                d1_items = ["Crossovers"] + mix.bands
                d1_sel = d1_items[self.indices[1]]
                if d1_sel == "Crossovers": return []
                modules = d1_sel.modules
                if not modules: return []
                module = modules[self.indices[2]]
                return list(module.parameters.values())
        except: return []
        return []

def test():
    initial_mixes = [Mix(crossovers=[100.0, 500.0, 3000.0]) for _ in range(5)]
    for m in initial_mixes:
        m.bands[0].modules.append(CompressorModule())
        m.bands[1].modules.append(TransientShaperModule())
        m.bands[2].modules.append(ExpanderModule())

    pop = Population(initial_mixes)
    state = MockUIState(pop)
    
    print("--- Initial State (Depth 0, Selection [0,0,0,0]) ---")
    print(f"Col 0 (Pop): {[f'Mix {i}' for i in range(len(state.get_items(0)))]}")
    print(f"Col 1 (Tree): {state.get_items(1)}")
    print(f"Col 2 (Modules/XO): {state.get_items(2)}")
    
    print("\n--- Move to Band 0 (Depth 1, Selection [0,1,0,0]) ---")
    state.indices[1] = 1 # Select Band: Low
    print(f"Col 1 (Tree) Selection: {state.get_items(1)[state.indices[1]]}")
    print(f"Col 2 (Modules): {state.get_items(2)}")
    
    if state.get_items(2):
        print("\n--- Move to Parameter Level (Depth 2, Selection [0,1,0,0]) ---")
        print(f"Col 3 (Params): {state.get_items(3)}")

if __name__ == "__main__":
    test()
