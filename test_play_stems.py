import sys
import os
import time
import threading
sys.path.append(os.path.abspath("."))
from core.mix import Mix
from audio.engine import Engine
import soundfile as sf
import sounddevice as sd
import numpy as np

def run():
    print("Loading test audio...")
    audio_data, sr = sf.read("test_audio.wav")
    audio_data = audio_data.T if audio_data.ndim > 1 else audio_data.reshape(1, -1)
    
    engine = Engine(sr)
    engine.update_mix(Mix(crossovers=[100, 500, 2000]))

    print("Separating...")
    stems = engine.separate_stems(audio_data, sr)

    stems_data = stems
    stem_names = ["vocals", "drums", "bass", "other"]
    stem_gains = {k: 0.0 for k in stem_names}
    stem_mutes = {k: False for k in stem_names}
    stem_solos = {k: False for k in stem_names}

    def _get_linear_gain(stem_name: str) -> float:
        db_gain = stem_gains.get(stem_name, 0.0)
        return 10 ** (db_gain / 20.0)

    play_idx = 0
    status_msg = ""
    
    def audio_callback(outdata, frames, time_info, status):
        nonlocal play_idx, status_msg
        
        audio_source = audio_data

        if stems_data is not None:
            try:
                min_len = min([stem.shape[1] for stem in stems_data.values()])
                num_channels = stems_data["vocals"].shape[0]
                pre_mixed = np.zeros((num_channels, min_len))

                for stem_name, stem_audio in stems_data.items():
                    is_muted = stem_mutes.get(stem_name, False)
                    is_soloed = stem_solos.get(stem_name, False)
                    any_solo = any(stem_solos.values())
                    
                    if any_solo and not is_soloed:
                        continue
                    if is_muted and not is_soloed:
                        continue

                    gain = _get_linear_gain(stem_name)
                    pre_mixed += stem_audio[:, :min_len] * gain

                if audio_data.shape[0] == 1 and num_channels > 1:
                    pre_mixed = np.mean(pre_mixed, axis=0, keepdims=True)

                audio_source = pre_mixed
            except Exception as e:
                status_msg = f"Stem mix error: {e}"
                audio_source = audio_data

        end_idx = play_idx + frames
        if end_idx <= audio_source.shape[1]:
            raw_chunk = audio_source[:, play_idx:end_idx]
            play_idx += frames
        else:
            n = audio_source.shape[1] - play_idx
            raw_chunk = np.zeros((audio_source.shape[0], frames))
            if n > 0:
                raw_chunk[:, :n] = audio_source[:, play_idx:]
            play_idx = 0

        try:
            processed_chunk = engine.process_chunk(raw_chunk)
            outdata[:] = processed_chunk.T
        except Exception as e:
            print("CRASH:", e)
            outdata.fill(0)

    stream = sd.OutputStream(
        samplerate=sr,
        channels=audio_data.shape[0],
        callback=audio_callback,
        blocksize=512
    )
    print("Starting stream...")
    stream.start()
    time.sleep(2)
    stream.stop()
    print("Stream done. Status msg:", status_msg)

run()
