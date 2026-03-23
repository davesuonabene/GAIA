import numpy as np
import librosa
from typing import Dict, Any, Optional
from dataclasses import dataclass
import sys
import os

# Ensure we can import from core when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from ..core.metadata import TrackMetadata
except (ImportError, ValueError):
    from core.metadata import TrackMetadata

@dataclass
class AnalysisConfig:
    """Configuration flags for the audio analyzer."""
    compute_rms: bool = True
    compute_transients: bool = True
    compute_spectral_centroid: bool = True
    compute_bpm: bool = False
    compute_mfcc: bool = False

class Analyzer:
    """Analyzes audio signals to extract key features based on provided configuration."""
    
    def __init__(self, config: AnalysisConfig = None, metadata: TrackMetadata = None):
        self.config = config or AnalysisConfig()
        self.metadata = metadata or TrackMetadata()

    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extracts configured features from the audio.
        """
        results = {}
        if len(audio) == 0:
            return results

        duration_sec = len(audio) / sample_rate

        # 1. Overall RMS (Loudness)
        if self.config.compute_rms:
            rms_frames = librosa.feature.rms(y=audio)
            results["rms"] = round(float(np.mean(rms_frames)), 6)

        # 2. Transient Density (Peak counts / Onset detection)
        if self.config.compute_transients:
            onsets = librosa.onset.onset_detect(y=audio, sr=sample_rate)
            transient_density = len(onsets) / duration_sec if duration_sec > 0 else 0.0
            results["transient_density"] = round(transient_density, 2)

        # 3. Spectral Centroid (Brightness)
        if self.config.compute_spectral_centroid:
            centroid_frames = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            results["spectral_centroid"] = round(float(np.mean(centroid_frames)), 2)

        # 4. BPM (Tempo)
        if self.config.compute_bpm:
            results["bpm"] = self._compute_bpm(audio, sample_rate)

        # 5. MFCC (Timbral texture)
        if self.config.compute_mfcc:
            results["mfcc"] = self._compute_mfcc(audio, sample_rate)

        return results

    def _compute_bpm(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Calculates the beats per minute (BPM) of the audio using librosa.
        """
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
            # librosa > 0.10 returns an array for tempo
            if isinstance(tempo, np.ndarray):
                return round(float(tempo[0]), 2)
            return round(float(tempo), 2)
        except Exception:
            return 0.0

    def _compute_mfcc(self, audio: np.ndarray, sample_rate: int) -> list:
        """
        Computes Mel-frequency cepstral coefficients (MFCCs).
        Returns the mean of each of the 13 coefficients across time frames.
        """
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            return [round(float(val), 4) for val in mfccs_mean]
        except Exception:
            return []

if __name__ == "__main__":
    import json
    
    # Test with standard config
    config = AnalysisConfig(compute_rms=True, compute_transients=True, compute_spectral_centroid=True)
    metadata = TrackMetadata(filename="test_burst.wav", sample_rate=44100, channels=1, duration_sec=1.0)
    analyzer = Analyzer(config=config, metadata=metadata)
    
    fs = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # --- TEST 1: Low-frequency Sine Wave (Bass) ---
    sine_wave = 0.5 * np.sin(2 * np.pi * 60 * t)
    bass_stats = analyzer.analyze(sine_wave, fs)
    
    # --- TEST 2: White Noise Burst (Snare-like) ---
    noise_burst = np.zeros(int(fs * duration))
    for i in range(5):
        start = int(i * 0.2 * fs)
        end = start + int(0.05 * fs)
        noise_burst[start:end] = np.random.uniform(-0.5, 0.5, end - start)
    
    snare_stats = analyzer.analyze(noise_burst, fs)
    
    print("=== AUDIO ANALYSIS TEST (CONFIG BASED) ===")
    print(f"Metadata: {metadata}")
    print(f"Config: {config}")
    
    print("\n[BASS SINE WAVE (60Hz)]")
    print(json.dumps(bass_stats, indent=2))
    
    print("\n[WHITE NOISE BURSTS (Transient/Bright)]")
    print(json.dumps(snare_stats, indent=2))
    
    # Validation Logic
    print("\n--- VERIFICATION ---")
    if snare_stats["spectral_centroid"] > bass_stats["spectral_centroid"]:
        print("PASS: Noise is brighter than Sine Wave.")
    else:
        print("FAIL: Centroid detection error.")
        
    if snare_stats["transient_density"] > bass_stats["transient_density"]:
        print("PASS: Noise bursts have more transients than smooth Sine Wave.")
    else:
        print("FAIL: Onset detection error.")
        
    # --- TEST 3: Config Flags ---
    print("\n--- TESTING CONFIG FLAGS ---")
    minimal_config = AnalysisConfig(compute_rms=True, compute_transients=False, compute_spectral_centroid=False)
    minimal_analyzer = Analyzer(config=minimal_config)
    minimal_stats = minimal_analyzer.analyze(noise_burst, fs)
    print("Minimal Stats (Only RMS):", json.dumps(minimal_stats))
    
    if "transient_density" not in minimal_stats and "spectral_centroid" not in minimal_stats:
        print("PASS: Feature exclusion works correctly.")
    else:
        print("FAIL: Unrequested features were computed.")
        
    # --- TEST 4: Full Features including BPM and MFCC ---
    print("\n--- TESTING FULL FEATURES ---")
    full_config = AnalysisConfig(
        compute_rms=True, 
        compute_transients=True, 
        compute_spectral_centroid=True,
        compute_bpm=True,
        compute_mfcc=True
    )
    full_analyzer = Analyzer(config=full_config)
    full_stats = full_analyzer.analyze(noise_burst, fs)
    print("Full Stats:", json.dumps(full_stats, indent=2))
    if "bpm" in full_stats and "mfcc" in full_stats and len(full_stats["mfcc"]) == 13:
        print("PASS: BPM and MFCC (13 coeffs) computed correctly.")
    else:
        print("FAIL: BPM or MFCC computation error.")