import numpy as np
import librosa
from typing import Dict, Any

class Analyzer:
    """Analyzes audio signals to extract key features for evolution feedback."""
    
    def analyze(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extracts RMS, Transient Density, and Spectral Centroid from the audio.
        """
        if len(audio) == 0:
            return {"rms": 0.0, "transient_density": 0.0, "spectral_centroid": 0.0}

        # 1. Overall RMS (Loudness)
        # librosa.feature.rms returns a frame-by-frame RMS; we take the mean.
        rms_frames = librosa.feature.rms(y=audio)
        rms = float(np.mean(rms_frames))

        # 2. Transient Density (Peak counts / Onset detection)
        # librosa.onset.onset_detect returns indices of detected onsets.
        onsets = librosa.onset.onset_detect(y=audio, sr=sample_rate)
        duration_sec = len(audio) / sample_rate
        transient_density = len(onsets) / duration_sec if duration_sec > 0 else 0.0

        # 3. Spectral Centroid (Brightness)
        # Higher values indicate a brighter sound (more high-frequency content).
        centroid_frames = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        spectral_centroid = float(np.mean(centroid_frames))

        return {
            "rms": round(rms, 6),
            "transient_density": round(transient_density, 2),
            "spectral_centroid": round(spectral_centroid, 2)
        }

if __name__ == "__main__":
    import json
    
    analyzer = Analyzer()
    fs = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # --- TEST 1: Low-frequency Sine Wave (Bass) ---
    # 60 Hz sine wave
    sine_wave = 0.5 * np.sin(2 * np.pi * 60 * t)
    bass_stats = analyzer.analyze(sine_wave, fs)
    
    # --- TEST 2: White Noise Burst (Snare-like) ---
    # 0.1s burst of white noise repeated 5 times to create clear transients
    noise_burst = np.zeros(int(fs * duration))
    for i in range(5):
        start = int(i * 0.2 * fs)
        end = start + int(0.05 * fs)
        noise_burst[start:end] = np.random.uniform(-0.5, 0.5, end - start)
    
    snare_stats = analyzer.analyze(noise_burst, fs)
    
    print("=== AUDIO ANALYSIS TEST ===")
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
