from typing import Dict, Any, Optional

import numpy as np
import os
import glob
import soundfile as sf
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
        if not pedalboard: return audio
        plugin = self.get_plugin(sample_rate)
        if plugin:
            return plugin.process(audio, sample_rate)
        return audio

    def get_plugin(self, sample_rate: int) -> Any:
        """Return a pedalboard plugin instance for this module. Should be overridden."""
        return None

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

    def get_plugin(self, sample_rate: int) -> Any:
        if not pedalboard: return None
        p = self.parameters
        
        # Prevent terrifying click/pop artifacts from near-0ms attack on bass-heavy material
        attack = max(2.0, p["Attack"].current_value)
        ratio = p["Ratio"].current_value
        threshold = p["Threshold"].current_value
        
        comp = pedalboard.Compressor(
            threshold_db=threshold,
            ratio=ratio,
            attack_ms=attack,
            release_ms=p["Release"].current_value,
        )
        
        # Auto makeup gain calculation for a gradual, predictable volume level
        # Approximates makeup for half of the maximum possible gain reduction
        max_gr = abs(min(0, threshold)) * (1.0 - (1.0 / ratio)) if ratio > 1.0 else 0.0
        auto_makeup = max_gr * 0.5
        
        gain = pedalboard.Gain(gain_db=p["Gain"].current_value + auto_makeup)
        return pedalboard.Pedalboard([comp, gain])

class ExpanderModule(AudioModule):
    """An Expander module implementation using pedalboard.NoiseGate."""
    def __init__(self):
        super().__init__("Expander")
        # Safe-spawn defaults (transparent)
        self.add_parameter(Parameter("Threshold", -80.0, -80.0, 0.0, 10.0))
        self.add_parameter(Parameter("Ratio", 1.0, 1.0, 10.0, 5.0))
        self.add_parameter(Parameter("Attack", 10.0, 0.1, 100.0, 10.0))
        self.add_parameter(Parameter("Release", 100.0, 10.0, 1000.0, 10.0))

    def get_plugin(self, sample_rate: int) -> Any:
        if not pedalboard: return None
        p = self.parameters
        return pedalboard.NoiseGate(
            threshold_db=p["Threshold"].current_value,
            ratio=p["Ratio"].current_value,
            attack_ms=p["Attack"].current_value,
            release_ms=p["Release"].current_value,
        )

class TransientShaperModule(AudioModule):
    """A compression-based Transient Shaper with parallel blending for extremely gradual scaling."""
    def __init__(self):
        super().__init__("TransientShaper")
        # Safe-spawn defaults (transparent)
        self.add_parameter(Parameter("Attack Boost", 0.0, -20.0, 20.0, 5.0))
        self.add_parameter(Parameter("Sustain Boost", 0.0, -20.0, 20.0, 5.0))

    def get_plugin(self, sample_rate: int) -> Any:
        return None

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not pedalboard or audio.size == 0: return audio
        
        att_db = self.parameters["Attack Boost"].current_value
        sus_db = self.parameters["Sustain Boost"].current_value
        
        if att_db == 0 and sus_db == 0:
            return audio
            
        current_audio = audio.copy()
        
        # Parallel blending ensures that at low values, the effect is nearly 100% dry, preventing volume jumps.
        if att_db != 0:
            if att_db > 0:
                # Boost transient: compress body (slow attack, fast release), blend with makeup
                comp = pedalboard.Compressor(threshold_db=-24.0, ratio=4.0, attack_ms=15.0, release_ms=50.0)
                wet_audio = comp.process(current_audio, sample_rate)
                mix = min(1.0, att_db / 20.0)
                makeup = 10 ** ((att_db * 0.5) / 20.0)
                current_audio = (current_audio * (1.0 - mix)) + (wet_audio * mix * makeup)
            else:
                # Cut transient: compress transient heavily (fast attack, fast release)
                comp = pedalboard.Compressor(threshold_db=-24.0, ratio=4.0, attack_ms=1.0, release_ms=50.0)
                wet_audio = comp.process(current_audio, sample_rate)
                mix = min(1.0, abs(att_db) / 20.0)
                current_audio = (current_audio * (1.0 - mix)) + (wet_audio * mix)
                
        if sus_db != 0:
            if sus_db > 0:
                # Boost sustain: compress transient (fast attack, slow release), blend with makeup
                comp = pedalboard.Compressor(threshold_db=-24.0, ratio=4.0, attack_ms=1.0, release_ms=250.0)
                wet_audio = comp.process(current_audio, sample_rate)
                mix = min(1.0, sus_db / 20.0)
                makeup = 10 ** ((sus_db * 0.5) / 20.0)
                current_audio = (current_audio * (1.0 - mix)) + (wet_audio * mix * makeup)
            else:
                # Cut sustain: expander on the tail
                gate = pedalboard.NoiseGate(threshold_db=-36.0, ratio=2.0, attack_ms=10.0, release_ms=100.0)
                wet_audio = gate.process(current_audio, sample_rate)
                mix = min(1.0, abs(sus_db) / 20.0)
                current_audio = (current_audio * (1.0 - mix)) + (wet_audio * mix)
                
        return current_audio

class SaturationModule(AudioModule):
    """A Saturation module with Drive and Mix parameters."""
    def __init__(self):
        super().__init__("Saturation")
        self.add_parameter(Parameter("Drive", 0.0, 0.0, 60.0, 10.0))
        self.add_parameter(Parameter("Mix", 100.0, 0.0, 100.0, 10.0))

    def get_plugin(self, sample_rate: int) -> Any:
        return None

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        drive_db = self.parameters["Drive"].current_value
        mix = self.parameters["Mix"].current_value / 100.0
        
        if drive_db == 0 or mix == 0:
            return audio
            
        # Smooth soft clipping (tanh) instead of digital harsh distortion
        drive_linear = 10 ** (drive_db / 20.0)
        
        # Apply tanh saturation. np.tanh is very smooth and analogous to analog saturation.
        wet_audio = np.tanh(audio * drive_linear)
        
        # Compensate volume to avoid huge spikes
        # The max amplitude is 1.0, but we need to normalize roughly
        wet_audio = wet_audio / np.maximum(1.0, np.log10(1.0 + drive_linear))
        
        return audio * (1.0 - mix) + wet_audio * mix

class ClipperModule(AudioModule):
    """A Clipper module implementation."""
    def __init__(self):
        super().__init__("Clipper")
        self.add_parameter(Parameter("Threshold", 0.0, -60.0, 0.0, 5.0))
        self.add_parameter(Parameter("Softness", 0.0, 0.0, 1.0, 0.1))

    def get_plugin(self, sample_rate: int) -> Any:
        if not pedalboard: return None
        try:
            return pedalboard.Clipping(threshold_db=self.parameters["Threshold"].current_value)
        except AttributeError:
            return None # Should handle in process() for fallback

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not pedalboard: return audio
        plugin = self.get_plugin(sample_rate)
        if plugin:
            return plugin.process(audio, sample_rate)
        # Fallback to a simple hard clip using numpy if pedalboard.Clipping is missing
        threshold_db = self.parameters["Threshold"].current_value
        threshold_linear = 10 ** (threshold_db / 20.0)
        return np.clip(audio, -threshold_linear, threshold_linear)

class LimiterModule(AudioModule):
    """A precise Limiter module implementation with Pre-Gain and Ceiling."""
    def __init__(self):
        super().__init__("Limiter")
        self.add_parameter(Parameter("Gain", 0.0, -24.0, 24.0, 5.0))
        self.add_parameter(Parameter("Ceiling", -0.1, -24.0, 0.0, 2.0))
        self.add_parameter(Parameter("Release", 100.0, 1.0, 1000.0, 10.0))

    def get_plugin(self, sample_rate: int) -> Any:
        if not pedalboard: return None
        p = self.parameters
        return pedalboard.Pedalboard([
            pedalboard.Gain(gain_db=p["Gain"].current_value),
            pedalboard.Limiter(
                threshold_db=p["Ceiling"].current_value,
                release_ms=p["Release"].current_value,
            )
        ])

class ConvolutionModule(AudioModule):
    """A Convolution module implementation with IR switching capabilities."""
    def __init__(self):
        super().__init__("Convolution")
        self.add_parameter(Parameter("Mix", 0.0, 0.0, 1.0, 0.1))
        self.add_parameter(Parameter("IR Index", 0.0, 0.0, 100.0, 1.0))

    def get_plugin(self, sample_rate: int) -> Any:
        if not pedalboard: return None
        
        os.makedirs("irs", exist_ok=True)
        ir_files = sorted(glob.glob("irs/*.wav"))
        
        ir = None
        if ir_files:
            idx = int(self.parameters["IR Index"].current_value) % len(ir_files)
            try:
                ir, ir_sr = sf.read(ir_files[idx])
                # Convert to mono if necessary or handle channels
                if ir.ndim > 1:
                    ir = np.mean(ir, axis=1)
                
                # Resample if mismatch? Pedalboard might handle it, but let's be safe
                # Actually pedalboard.Convolution takes a sample_rate param
            except Exception:
                ir = None

        if ir is None:
            # Fallback to the synthetic Dirac impulse
            ir = np.zeros(100, dtype=np.float32)
            ir[0] = 1.0
        else:
            ir = ir.astype(np.float32)
        
        return pedalboard.Convolution(ir, mix=self.parameters["Mix"].current_value, sample_rate=sample_rate)


if __name__ == "__main__":
    # --- SAFE SPAWN VERIFICATION ---
    print("=== INITIAL SAFE-SPAWN STATES (DNA TRANSPARENCY) ===")
    
    comp = CompressorModule()
    exp = ExpanderModule()
    ts = TransientShaperModule()
    
    for module in [comp, exp, ts]:
        print(f"\n{module}")
