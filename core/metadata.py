from dataclasses import dataclass
from typing import Optional

@dataclass
class TrackMetadata:
    """Holds fundamental metadata about an audio track."""
    filename: str = ""
    sample_rate: int = 0
    channels: int = 0
    duration_sec: float = 0.0
    bpm: Optional[float] = None
    key: Optional[str] = None

if __name__ == "__main__":
    # Simple smoke test
    meta = TrackMetadata(filename="test.wav", sample_rate=44100, channels=2, duration_sec=120.5)
    print("Default Metadata:", TrackMetadata())
    print("Test Metadata:", meta)