import time, soundfile as sf
import numpy as np
import sounddevice as sd
import threading

data, sr = sf.read("test_audio.wav")
audio = data.T if data.ndim > 1 else data.reshape(1, -1)
print(audio.shape)
# Make it 10 seconds long by tiling
audio = np.tile(audio, (1, 10))
processed = audio.copy()

idx = 0
def cb(outdata, frames, time, status):
    global idx, processed
    if status:
        print(status)
    end_idx = min(idx + frames, processed.shape[1])
    n = end_idx - idx
    outdata[:n, :] = processed[:, idx:end_idx].T
    if n < frames:
        outdata[n:, :] = 0
        raise sd.CallbackStop()
    idx += frames

stream = sd.OutputStream(samplerate=sr, channels=audio.shape[0], callback=cb)
stream.start()

time.sleep(2)
print("Modifying processed audio")
processed[:] = processed * 0.1 # Volume drop
time.sleep(2)
stream.stop()
