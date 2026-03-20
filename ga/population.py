import copy
from typing import List

try:
    from ..core.mix import Mix
except (ImportError, ValueError):
    from core.mix import Mix

class Population:
    """Manages a collection of Mix candidates (the population) for the GA."""
    def __init__(self, mixes: List[Mix]):
        self.mixes = mixes
        self.generation_count = 0

    def generate_next_generation(self, 
                                 parent_mix_index: int, 
                                 structural_rate: float, 
                                 parametric_rate: float, 
                                 batch_size: int = 5) -> None:
        """
        Creates a new generation based on a single selected parent.
        Applies cloning and mutations.
        """
        if not (0 <= parent_mix_index < len(self.mixes)):
            raise ValueError(f"Parent index {parent_mix_index} is out of bounds.")
        
        parent = self.mixes[parent_mix_index]
        new_mixes = []
        
        # In a real GA, we might keep the parent as-is (elitism)
        # For Phase 5, we spawn 'batch_size' children from this parent.
        for _ in range(batch_size):
            # 1. Clone the parent DNA
            child = copy.deepcopy(parent)
            
            # 2. Mutate the clone
            child.evolve(structural_mutation_rate=structural_rate, 
                         parametric_mutation_rate=parametric_rate)
            
            new_mixes.append(child)
            
        self.mixes = new_mixes
        self.generation_count += 1

    def __repr__(self) -> str:
        return f"Population | Gen: {self.generation_count} | Size: {len(self.mixes)}"

if __name__ == "__main__":
    # --- POPULATION BATCH TEST ---
    from core.mix import Mix
    from core.audio_module import CompressorModule

    # 1. Initialize Generation 0 with 5 empty mixes (2 crossovers)
    initial_mixes = [Mix(crossovers=[150.0, 2500.0]) for _ in range(5)]
    
    # Let's seed the parent (Mix 0) with at least one module so we can see it mutate
    initial_mixes[0].bands[0].modules.append(CompressorModule())
    
    pop = Population(initial_mixes)
    print(f"=== {pop} ===")
    print(f"Initial Parent (Mix 0) structure: {pop.mixes[0].bands[0]}")
    
    # 2. Programmatically select Mix 0 as the parent and spawn Gen 1
    # Use high mutation rates to ensure visible changes
    print("\n--- Spawning Generation 1 (Parent = Mix 0) ---")
    pop.generate_next_generation(parent_mix_index=0, 
                                 structural_rate=0.8, 
                                 parametric_rate=0.8, 
                                 batch_size=5)
    
    print(f"=== {pop} ===")
    
    # 3. Compare DNA
    print("\nComparison of Children:")
    for i, child in enumerate(pop.mixes):
        print(f"Child {i}: {child.bands[0]}")
        # Show first module parameters for Child 0 as a sample
        if i == 0 and child.bands[0].modules:
            mod = child.bands[0].modules[0]
            print(f"  Sample {mod.name} Param (Thresh): {mod.parameters['Threshold'].current_value:.2f}")
