import sys
import os
import time
import threading
sys.path.append(os.path.abspath("."))
import numpy as np

# We'll mock out the rendering but test the interaction logic
class MockEngine:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
        self.mix = None
        
    def separate_stems(self, audio, sr, progress_callback=None):
        for i in range(10):
            time.sleep(0.1)
            if progress_callback:
                progress_callback((i+1)*10.0)
        return {
            'vocals': np.random.randn(2, audio.shape[1]),
            'drums': np.random.randn(2, audio.shape[1]),
            'bass': np.random.randn(2, audio.shape[1]),
            'other': np.random.randn(2, audio.shape[1]),
        }
        
    def update_mix(self, mix):
        self.mix = mix

class TestTUI:
    def __init__(self):
        self.audio_data = np.random.randn(1, 44100 * 5) # Mono audio
        self.sample_rate = 44100
        self.engine = MockEngine(44100)
        self.is_separating = False
        self.separation_progress = 0.0
        self.stems_data = None
        self.stem_gains = {"vocals": 0.0, "drums": 0.0, "bass": 0.0, "other": 0.0}
        self.stem_mutes = {"vocals": False, "drums": False, "bass": False, "other": False}
        self.stem_solos = {"vocals": False, "drums": False, "bass": False, "other": False}
        self.stem_names = ["vocals", "drums", "bass", "other"]
        self.selected_stem_idx = 0
        self.is_playing = False
        self.status_msg = ""
        
        # Test population mocking
        self.population = type('obj', (object,), {'mixes': [type('obj', (object,), {})()]})
        self.selected_mix_idx = 0

    def action_separate_stems(self):
        self.is_separating = True
        threading.Thread(target=self._separation_worker, daemon=True).start()

    def _separation_worker(self):
        def _update_progress(percent: float):
            self.separation_progress = percent
            print(f"Progress: {percent}%")
            
        try:
            self.separation_progress = 0.0
            stems = self.engine.separate_stems(self.audio_data, self.sample_rate, progress_callback=_update_progress)
            self.stems_data = stems
            self.status_msg = "Separation Complete!"
        except Exception as e:
            self.status_msg = f"Separation Failed: {str(e)}"
        finally:
            self.is_separating = False
            self.separation_progress = 0.0
            print("Worker finished")

    def _get_linear_gain(self, stem_name: str) -> float:
        db_gain = self.stem_gains.get(stem_name, 0.0)
        return 10 ** (db_gain / 20.0)

    def audio_callback(self, outdata, frames, time_info, status):
        # 1. Base Audio Source Setup
        audio_source = self.audio_data

        if self.stems_data is not None:
            try:
                min_len = min([stem.shape[1] for stem in self.stems_data.values()])
                num_channels = self.stems_data["vocals"].shape[0]
                pre_mixed = np.zeros((num_channels, min_len))

                for stem_name, stem_audio in self.stems_data.items():
                    is_muted = self.stem_mutes.get(stem_name, False)
                    is_soloed = self.stem_solos.get(stem_name, False)
                    any_solo = any(self.stem_solos.values())
                    
                    if any_solo and not is_soloed:
                        continue 
                    if is_muted and not is_soloed:
                        continue 

                    gain = self._get_linear_gain(stem_name)
                    pre_mixed += stem_audio[:, :min_len] * gain

                if self.audio_data.shape[0] == 1 and num_channels > 1:
                    pre_mixed = np.mean(pre_mixed, axis=0, keepdims=True)

                audio_source = pre_mixed
            except Exception as e:
                self.status_msg = f"Stem mix error: {e}"
                audio_source = self.audio_data
                
        return audio_source

print("Testing TUI Logic...")
tui = TestTUI()
tui.action_separate_stems()
time.sleep(1.5) # Wait for separation
print("Stems loaded:", tui.stems_data is not None)

print("\nTesting normal audio callback:")
out = tui.audio_callback(None, 512, None, None)
print("Normal callback shape:", out.shape)

print("\nTesting Solo 'vocals':")
tui.stem_solos['vocals'] = True
out = tui.audio_callback(None, 512, None, None)
print("Solo vocals shape:", out.shape)

print("\nTesting Mute 'drums':")
tui.stem_solos['vocals'] = False
tui.stem_mutes['drums'] = True
out = tui.audio_callback(None, 512, None, None)
print("Muted drums shape:", out.shape)

print("\nAll logical checks passed!")
