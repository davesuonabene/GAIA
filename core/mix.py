from typing import List

try:
    from .band import Band
except (ImportError, ValueError):
    from band import Band

class Mix:
    """A multi-band audio mix processing graph."""
    def __init__(self, crossovers: List[float]):
        """
        Initializes a Mix with the given crossover frequencies.
        The number of bands created will be len(crossovers) + 1.
        """
        self.crossovers = sorted(crossovers)
        self.bands: List[Band] = []
        
        # Create band names (Low, Mid, High, or numerical)
        band_names = ["Low", "Mid", "High"]
        if len(self.crossovers) + 1 > 3:
            band_names = [f"Band {i+1}" for i in range(len(self.crossovers) + 1)]
            
        for i in range(len(self.crossovers) + 1):
            name = band_names[i] if i < len(band_names) else f"Band {i+1}"
            self.bands.append(Band(name))

    def evolve(self, structural_mutation_rate: float = 0.5, parametric_mutation_rate: float = 0.5) -> None:
        """Evolves the mix by applying structural and parameter mutations to all bands."""
        for band in self.bands:
            band.mutate_structure(structural_mutation_rate)
            band.mutate_parameters(parametric_mutation_rate)

    def __repr__(self) -> str:
        mix_info = f"Mix Graph | Crossovers: {self.crossovers} Hz\n"
        band_info = "\n".join([f"  {band}" for band in self.bands])
        return mix_info + band_info

if __name__ == "__main__":
    # --- SIMULATION SETUP ---
    # Create a 3-band mix (requires 2 crossover points)
    my_mix = Mix(crossovers=[150.0, 2500.0])
    
    print("=== INITIAL GENERATION ===")
    print(my_mix)
    print("-" * 40)
    
    # Run a simulation for 3 generations
    for gen in range(1, 4):
        # Use a high mutation rate to observe frequent changes in structure
        my_mix.evolve(structural_mutation_rate=0.8)
        
        print(f"\n=== GENERATION {gen} ===")
        print(my_mix)
        
        # Display internal parameter values for the first band to confirm drift
        if my_mix.bands[0].modules:
             first_mod = my_mix.bands[0].modules[0]
             print(f"  [Sample Debug: {first_mod.name} in Low Band]")
             for p in list(first_mod.parameters.values())[:2]: # Show first 2 params
                 print(f"    - {p}")
        
        print("-" * 40)
