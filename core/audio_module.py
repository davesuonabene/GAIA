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

    def to_dict(self) -> dict:
        """Serialize audio module to a dictionary."""
        return {
            "module_type": self.__class__.__name__,
            "parameters": {name: p.to_dict() for name, p in self.parameters.items()}
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Reconstruct audio module from a dictionary."""
        # Find the correct class in the module's globals if not specified
        module_cls = cls
        if "module_type" in data:
            module_cls = globals().get(data["module_type"], cls)
        
        instance = module_cls()
        for p_name, p_data in data.get("parameters", {}).items():
            if p_name in instance.parameters:
                instance.parameters[p_name] = Parameter.from_dict(p_data)
        return instance

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

class ClipperModule(AudioModule):
    """A Clipper module implementation."""
    def __init__(self):
        super().__init__("Clipper")
        self.add_parameter(Parameter("Threshold", 0.0, -60.0, 0.0, 5.0))
        self.add_parameter(Parameter("Softness", 0.0, 0.0, 1.0, 0.1))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not pedalboard: return audio
        # Using pedalboard.Clipping if available, otherwise Distortion
        try:
            clipper = pedalboard.Clipping(threshold_db=self.parameters["Threshold"].current_value)
            return clipper.process(audio, sample_rate)
        except AttributeError:
            # Fallback to a simple hard clip using numpy if pedalboard.Clipping is missing
            threshold_db = self.parameters["Threshold"].current_value
            threshold_linear = 10 ** (threshold_db / 20.0)
            return np.clip(audio, -threshold_linear, threshold_linear)

class LimiterModule(AudioModule):
    """A Limiter module implementation."""
    def __init__(self):
        super().__init__("Limiter")
        self.add_parameter(Parameter("Threshold", 0.0, -60.0, 0.0, 5.0))
        self.add_parameter(Parameter("Release", 100.0, 1.0, 1000.0, 10.0))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not pedalboard: return audio
        p = self.parameters
        limiter = pedalboard.Limiter(
            threshold_db=p["Threshold"].current_value,
            release_ms=p["Release"].current_value,
        )
        return limiter.process(audio, sample_rate)

class ConvolutionModule(AudioModule):
    """A Convolution module implementation."""
    def __init__(self):
        super().__init__("Convolution")
        self.add_parameter(Parameter("Mix", 0.0, 0.0, 1.0, 0.1))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not pedalboard: return audio
        # Default IR: a single impulse (no-op)
        ir = np.zeros(100)
        ir[0] = 1.0
        conv = pedalboard.Convolution(ir, mix=self.parameters["Mix"].current_value)
        return conv.process(audio, sample_rate)


if __name__ == "__main__":
    # --- SAFE SPAWN VERIFICATION ---
    print("=== INITIAL SAFE-SPAWN STATES (DNA TRANSPARENCY) ===")
    
    comp = CompressorModule()
    exp = ExpanderModule()
    ts = TransientShaperModule()
    
    for module in [comp, exp, ts]:
        print(f"\n{module}")
