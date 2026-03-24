import numpy as np

# Test if mute/solo logic impacts pre_mixed audio shape or playback
stems_data = {
    'vocals': np.random.randn(2, 44100 * 2), 
    'drums': np.random.randn(2, 44100 * 2),
    'bass': np.random.randn(2, 44100 * 2),
    'other': np.random.randn(2, 44100 * 2)
}

stem_gains = {'vocals': 0.0, 'drums': -60.0, 'bass': 0.0, 'other': 0.0} # simulate a mute
audio_data = np.random.randn(2, 44100 * 2)

def _get_linear_gain(stem_name: str) -> float:
    db_gain = stem_gains.get(stem_name, 0.0)
    return 10 ** (db_gain / 20.0)

try:
    min_len = min([stem.shape[1] for stem in stems_data.values()])
    pre_mixed = np.zeros((audio_data.shape[0], min_len))

    for stem_name, stem_audio in stems_data.items():
        gain = _get_linear_gain(stem_name)
        pre_mixed += stem_audio[:, :min_len] * gain

    audio_source = pre_mixed
    print("Success. Shape:", audio_source.shape)
except Exception as e:
    print("Error:", e)
