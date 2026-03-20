import random
from typing import List

try:
    from .audio_module import AudioModule, CompressorModule
except (ImportError, ValueError):
    from audio_module import AudioModule, CompressorModule

class Band:
    """A collection of AudioModules processing a specific frequency range."""
    def __init__(self, name: str):
        self.name = name
        self.modules: List[AudioModule] = []

    def mutate_structure(self, mutation_rate: float) -> None:
        """
        Randomly add, remove, or swap modules in the chain based on mutation_rate (0.0 to 1.0).
        """
        # Add a module
        if random.random() < mutation_rate:
            # For now, we only have CompressorModule
            new_module = CompressorModule()
            self.modules.append(new_module)

        # Remove a module
        if self.modules and random.random() < mutation_rate:
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

    def __repr__(self) -> str:
        if not self.modules:
            return f"Band: {self.name} (Empty)"
        
        module_chain = " -> ".join([m.name for m in self.modules])
        return f"Band: {self.name} | Chain: {module_chain}"
