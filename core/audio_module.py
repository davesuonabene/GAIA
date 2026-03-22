from typing import Dict

import numpy as np
try:
    import pedalboard
except ImportError:
    pedalboard = None

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

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio with this module. Should be overridden by subclasses."""
        return audio

    def __repr__(self) -> str:
        param_str = "\n".join([f"  - {p}" for p in self.parameters.values()])
        return f"AudioModule: {self.name}\n{param_str}"

class CompressorModule(AudioModule):
    """A Compressor module implementation."""
    def __init__(self):
        super().__init__("Compressor")
        # Initialize with 'safe-spawning' defaults
        self.add_parameter(Parameter("Threshold", 0.0, -60.0, 0.0, 10.0))
        self.add_parameter(Parameter("Ratio", 1.0, 1.0, 20.0, 5.0))
        self.add_parameter(Parameter("Attack", 10.0, 0.1, 500.0, 10.0))
        self.add_parameter(Parameter("Release", 100.0, 1.0, 2000.0, 10.0))
        self.add_parameter(Parameter("Gain", 0.0, -12.0, 24.0, 5.0))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not pedalboard: return audio
        p = self.parameters
        comp = pedalboard.Compressor(
            threshold_db=p["Threshold"].current_value,
            ratio=p["Ratio"].current_value,
            attack_ms=p["Attack"].current_value,
            release_ms=p["Release"].current_value,
        )
        gain = pedalboard.Gain(gain_db=p["Gain"].current_value)
        board = pedalboard.Pedalboard([comp, gain])
        return board.process(audio, sample_rate)

class ExpanderModule(AudioModule):
    """An Expander module implementation using pedalboard.NoiseGate."""
    def __init__(self):
        super().__init__("Expander")
        # Safe-spawn defaults (transparent)
        self.add_parameter(Parameter("Threshold", -80.0, -80.0, 0.0, 10.0))
        self.add_parameter(Parameter("Ratio", 1.0, 1.0, 10.0, 5.0))
        self.add_parameter(Parameter("Attack", 10.0, 0.1, 100.0, 10.0))
        self.add_parameter(Parameter("Release", 100.0, 10.0, 1000.0, 10.0))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not pedalboard: return audio
        p = self.parameters
        gate = pedalboard.NoiseGate(
            threshold_db=p["Threshold"].current_value,
            ratio=p["Ratio"].current_value,
            attack_ms=p["Attack"].current_value,
            release_ms=p["Release"].current_value,
        )
        return gate.process(audio, sample_rate)

class TransientShaperModule(AudioModule):
    """A Transient Shaper module implementation (Approximation)."""
    def __init__(self):
        super().__init__("TransientShaper")
        # Safe-spawn defaults (transparent)
        self.add_parameter(Parameter("Attack Boost", 0.0, -20.0, 20.0, 5.0))
        self.add_parameter(Parameter("Sustain Boost", 0.0, -20.0, 20.0, 5.0))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not pedalboard: return audio
        # pedalboard doesn't have a transient shaper. 
        # For now, we use a simple Gain scaling as an approximation.
        att = self.parameters["Attack Boost"].current_value
        sus = self.parameters["Sustain Boost"].current_value
        gain = pedalboard.Gain(gain_db=(att + sus) / 2.0)
        return gain.process(audio, sample_rate)

class SaturationModule(AudioModule):
    """A Saturation module with Drive and Mix parameters."""
    def __init__(self):
        super().__init__("Saturation")
        self.add_parameter(Parameter("Drive", 0.0, 0.0, 60.0, 10.0))
        self.add_parameter(Parameter("Mix", 100.0, 0.0, 100.0, 10.0))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not pedalboard: return audio
        drive = self.parameters["Drive"].current_value
        mix = self.parameters["Mix"].current_value / 100.0
        dist = pedalboard.Distortion(drive_db=drive)
        distorted = dist.process(audio, sample_rate)
        return (mix * distorted) + ((1.0 - mix) * audio)


if __name__ == "__main__":
    # --- SAFE SPAWN VERIFICATION ---
    print("=== INITIAL SAFE-SPAWN STATES (DNA TRANSPARENCY) ===")
    
    comp = CompressorModule()
    exp = ExpanderModule()
    ts = TransientShaperModule()
    
    for module in [comp, exp, ts]:
        print(f"\n{module}")
