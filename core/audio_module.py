from typing import Dict

try:
    from .parameter import Parameter
except (ImportError, ValueError):
    from parameter import Parameter

class AudioModule:
    """Base class for all audio processing modules."""
    def __init__(self, name: str):
        self.name = name
        self.parameters: Dict[str, Parameter] = {}

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the module."""
        self.parameters[param.name] = param

    def mutate_parameters(self, rate: float = 1.0) -> None:
        """Call mutate on all parameters in the module based on mutation rate."""
        for param in self.parameters.values():
            param.mutate(rate)

    def __repr__(self) -> str:
        param_str = "\n".join([f"  - {p}" for p in self.parameters.values()])
        return f"AudioModule: {self.name}\n{param_str}"

class CompressorModule(AudioModule):
    """A dummy Compressor module implementation."""
    def __init__(self):
        super().__init__("Compressor")
        # Initialize with 'safe-spawning' defaults
        self.add_parameter(Parameter("Threshold", 0.0, -60.0, 0.0, 10.0))
        self.add_parameter(Parameter("Ratio", 1.0, 1.0, 20.0, 5.0))
        self.add_parameter(Parameter("Attack", 10.0, 0.1, 500.0, 10.0))
        self.add_parameter(Parameter("Release", 100.0, 1.0, 2000.0, 10.0))
        self.add_parameter(Parameter("Gain", 0.0, -12.0, 24.0, 5.0))

class ExpanderModule(AudioModule):
    """An Expander module implementation."""
    def __init__(self):
        super().__init__("Expander")
        # Safe-spawn defaults (transparent)
        self.add_parameter(Parameter("Threshold", -80.0, -80.0, 0.0, 10.0))
        self.add_parameter(Parameter("Ratio", 1.0, 1.0, 10.0, 5.0))
        self.add_parameter(Parameter("Attack", 10.0, 0.1, 100.0, 10.0))
        self.add_parameter(Parameter("Release", 100.0, 10.0, 1000.0, 10.0))

class TransientShaperModule(AudioModule):
    """A Transient Shaper module implementation."""
    def __init__(self):
        super().__init__("TransientShaper")
        # Safe-spawn defaults (transparent)
        self.add_parameter(Parameter("Attack Boost", 0.0, -100.0, 100.0, 15.0))
        self.add_parameter(Parameter("Sustain Boost", 0.0, -100.0, 100.0, 15.0))

if __name__ == "__main__":
    # --- SAFE SPAWN VERIFICATION ---
    print("=== INITIAL SAFE-SPAWN STATES (DNA TRANSPARENCY) ===")
    
    comp = CompressorModule()
    exp = ExpanderModule()
    ts = TransientShaperModule()
    
    for module in [comp, exp, ts]:
        print(f"\n{module}")
