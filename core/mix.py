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
        self.stem_bands = {
            "vocals": Band("Vocals"),
            "drums": Band("Drums"),
            "bass": Band("Bass"),
            "other": Band("Other")
        }
        self.pre_band = Band("PRE")
        self.post_band = Band("POST")
        
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
            
        self.is_locked = False

    def to_dict(self) -> dict:
        """Serialize mix to a dictionary."""
        return {
            "stem_bands": {k: v.to_dict() for k, v in self.stem_bands.items()},
            "crossover_params": [p.to_dict() for p in self.crossover_params],
            "pre_band": self.pre_band.to_dict(),
            "bands": [b.to_dict() for b in self.bands],
            "post_band": self.post_band.to_dict(),
            "is_locked": self.is_locked
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Reconstruct mix from a dictionary."""
        # Create a basic Mix instance with dummy crossovers
        instance = cls(crossovers=[])
        
        if "stem_bands" in data:
            instance.stem_bands = {k: Band.from_dict(v) for k, v in data["stem_bands"].items()}
        
        if "crossover_params" in data:
            instance.crossover_params = [Parameter.from_dict(p) for p in data["crossover_params"]]
        
        if "pre_band" in data:
            instance.pre_band = Band.from_dict(data["pre_band"])
        
        if "bands" in data:
            instance.bands = [Band.from_dict(b) for b in data["bands"]]
            
        if "post_band" in data:
            instance.post_band = Band.from_dict(data["post_band"])
            
        instance.is_locked = data.get("is_locked", False)
            
        return instance

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

    def sort_crossovers(self) -> None:
        """Sorts the crossover parameters by frequency and updates their names to maintain logical order."""
        self.crossover_params.sort(key=lambda p: p.current_value)
        for i, p in enumerate(self.crossover_params):
            p.name = f"Crossover {i+1}"

    def evolve(self, structural_mutation_rate: float = 0.5, parametric_mutation_rate: float = 0.5) -> None:
        """Evolves the mix by applying structural and parameter mutations."""
        # 1. Mutate crossover frequencies
        for p in self.crossover_params:
            p.mutate(parametric_mutation_rate)
        
        # Sort and rename crossover parameters to maintain band order
        self.sort_crossovers()
        
        # 2. Mutate stem bands
        for band in self.stem_bands.values():
            band.mutate_structure(structural_mutation_rate)
            band.mutate_parameters(parametric_mutation_rate)

        # 3. Mutate bands (PRE, POST, and individual frequency bands)
        for band in [self.pre_band] + self.bands + [self.post_band]:
            band.mutate_structure(structural_mutation_rate)
            band.mutate_parameters(parametric_mutation_rate)

    def __repr__(self) -> str:
        c_info = ", ".join([f"{f:.1f}" for f in self.crossover_frequencies])
        mix_info = f"Mix Graph | Crossovers: [{c_info}] Hz\n"
        pre_info = f"  {self.pre_band}\n"
        band_info = "\n".join([f"  {band}" for band in self.bands])
        post_info = f"\n  {self.post_band}"
        return mix_info + pre_info + band_info + post_info

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
