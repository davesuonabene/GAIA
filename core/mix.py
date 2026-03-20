from typing import List

try:
    from .band import Band
    from .parameter import Parameter
except (ImportError, ValueError):
    from band import Band
    from parameter import Parameter

class Mix:
    """A multi-band audio mix processing graph."""
    def __init__(self, crossovers: List[float]):
        """
        Initializes a Mix with the given crossover frequencies.
        Crossovers are stored as evolvable Parameter objects.
        """
        self.crossover_params: List[Parameter] = []
        for i, freq in enumerate(sorted(crossovers)):
            # Crossovers can range from 20Hz to 20kHz, with a 10% drift range.
            self.crossover_params.append(
                Parameter(f"Crossover {i+1}", freq, 20.0, 20000.0, 10.0)
            )
            
        self.bands: List[Band] = []
        
        # Create band names (Low, Mid, High, or numerical)
        band_names = ["Low", "Mid", "High"]
        num_bands = len(self.crossover_params) + 1
        if num_bands > 3:
            band_names = [f"Band {i+1}" for i in range(num_bands)]
            
        for i in range(num_bands):
            name = band_names[i] if i < len(band_names) else f"Band {i+1}"
            self.bands.append(Band(name))

    @property
    def crossover_frequencies(self) -> List[float]:
        """Returns a sorted list of current crossover frequency values."""
        # Sorting ensures that if parameters drift past each other, bands stay logical.
        freqs = sorted([p.current_value for p in self.crossover_params])
        return freqs

    @property
    def crossovers(self) -> List[float]:
        """Alias for crossover_frequencies to maintain compatibility with the engine."""
        return self.crossover_frequencies

    def evolve(self, structural_mutation_rate: float = 0.5, parametric_mutation_rate: float = 0.5) -> None:
        """Evolves the mix by applying structural and parameter mutations."""
        # 1. Mutate crossover frequencies
        for p in self.crossover_params:
            p.mutate(parametric_mutation_rate)
        
        # Sort crossover parameters by their current value to maintain band order
        self.crossover_params.sort(key=lambda p: p.current_value)
        
        # 2. Mutate bands
        for band in self.bands:
            band.mutate_structure(structural_mutation_rate)
            band.mutate_parameters(parametric_mutation_rate)

    def __repr__(self) -> str:
        c_info = ", ".join([f"{f:.1f}" for f in self.crossover_frequencies])
        mix_info = f"Mix Graph | Crossovers: [{c_info}] Hz\n"
        band_info = "\n".join([f"  {band}" for band in self.bands])
        return mix_info + band_info

if __name__ == "__main__":
    # --- CROSSOVER EVOLUTION TEST ---
    print("=== CROSSOVER EVOLUTION TEST ===")
    
    # Create a Mix with crossovers [100.0, 1000.0]
    test_mix = Mix(crossovers=[100.0, 1000.0])
    
    # Lock the parameter named "Crossover 1"
    locked_param = next(p for p in test_mix.crossover_params if p.name == "Crossover 1")
    locked_param.is_locked = True
    
    print(f"Initial State:\n{test_mix}")
    print("-" * 40)
    
    # Run evolve() 5 times with a 1.0 parametric mutation rate
    for gen in range(1, 6):
        test_mix.evolve(structural_mutation_rate=0.0, parametric_mutation_rate=1.0)
        
        # Find parameters by name as their index might change due to sorting
        c1 = next(p for p in test_mix.crossover_params if p.name == "Crossover 1")
        c2 = next(p for p in test_mix.crossover_params if p.name == "Crossover 2")
        
        print(f"Generation {gen}:")
        print(f"  {c1.name} (LOCKED):   {c1.current_value:7.2f} Hz")
        print(f"  {c2.name} (UNLOCKED): {c2.current_value:7.2f} Hz")
        print(f"  Sorted Crossovers: {test_mix.crossover_frequencies}")
        print("-" * 40)
        
        # Verification
        assert abs(c1.current_value - 100.0) < 1e-6, f"{c1.name} changed!"
    
    print("TEST PASSED: Locked crossover stayed at exactly 100.0Hz while the other drifted.")
