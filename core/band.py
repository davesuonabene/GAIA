import random
from typing import List

try:
    from .audio_module import (
        AudioModule, CompressorModule, ExpanderModule, 
        TransientShaperModule, SaturationModule, ClipperModule, 
        LimiterModule, ConvolutionModule
    )
    from .parameter import Parameter
except (ImportError, ValueError):
    from audio_module import (
        AudioModule, CompressorModule, ExpanderModule, 
        TransientShaperModule, SaturationModule, ClipperModule, 
        LimiterModule, ConvolutionModule
    )
    from parameter import Parameter

class Band:
    """A collection of AudioModules processing a specific frequency range."""
    def __init__(self, name: str):
        self.name = name
        self.modules: List[AudioModule] = []
        self.is_muted = False
        self.is_soloed = False
        self.gain = Parameter("Gain", 0.0, -24.0, 24.0, 10.0)

    def mutate_structure(self, mutation_rate: float) -> None:
        """
        Randomly add, remove, or swap modules in the chain based on mutation_rate (0.0 to 1.0).
        Enforces a maximum of one instance per module class.
        """
        # Module pool for expansion
        module_pool = [
            CompressorModule, ExpanderModule, TransientShaperModule, 
            SaturationModule, ClipperModule, LimiterModule, ConvolutionModule
        ]

        # Add a module
        if random.random() < mutation_rate:
            # Filter pool for modules not already present
            existing_classes = {type(m) for m in self.modules}
            available_pool = [cls for cls in module_pool if cls not in existing_classes]
            
            if available_pool:
                new_module_cls = random.choice(available_pool)
                self.modules.append(new_module_cls())

        # Remove a module
        # Using a slight bias for removal to allow the chain to stabilize
        if self.modules and random.random() < (mutation_rate * 0.5):
            index = random.randrange(len(self.modules))
            self.modules.pop(index)

        # Swap modules
        if len(self.modules) > 1 and random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(self.modules)), 2)
            self.modules[idx1], self.modules[idx2] = self.modules[idx2], self.modules[idx1]

    def mutate_parameters(self, rate: float = 1.0) -> None:
        """Trigger parameter drift for all modules in the band based on mutation rate."""
        for module in self.modules:
            module.mutate_parameters(rate)
        self.gain.mutate(rate)

    def __repr__(self) -> str:
        state = ""
        if self.is_soloed: state += "[S]"
        if self.is_muted: state += "[M]"
        
        module_chain = " -> ".join([m.name for m in self.modules]) if self.modules else "Empty"
        return f"Band: {self.name} {state} | Gain: {self.gain.current_value:.1f}dB | Chain: {module_chain}"

if __name__ == "__main__":
    # --- STRUCTURAL MUTATION TEST ---
    print("=== MULTI-MODULE STRUCTURAL EVOLUTION TEST ===")
    
    test_band = Band("Drum Bus")
    print(f"Initial: {test_band}")
    
    # Run 10 mutations with mutation_rate of 1.0.
    for i in range(1, 11):
        test_band.mutate_structure(mutation_rate=1.0)
        print(f"Mutation #{i:2d}: {test_band}")
    
    print("\nFinal evolved chain verification:")
    if test_band.modules:
        print(f"Chain length: {len(test_band.modules)}")
        for i, mod in enumerate(test_band.modules):
            print(f"  Slot {i}: {mod.name}")
    else:
        print("Band is empty (random removal cleared it).")
