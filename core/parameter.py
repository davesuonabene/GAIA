import random
from dataclasses import dataclass

@dataclass
class Parameter:
    name: str
    current_value: float
    min_bound: float
    max_bound: float
    drift_range: float  # Drift range as a percentage of the total parameter range
    is_locked: bool = False

    def mutate(self, rate: float = 1.0) -> None:
        """Apply a random drift to the parameter if it's not locked and based on rate."""
        if self.is_locked or random.random() > rate:
            return
        
        total_range = self.max_bound - self.min_bound
        max_delta = total_range * (self.drift_range / 100.0)
        
        # Apply random drift within [-max_delta, max_delta]
        delta = random.uniform(-max_delta, max_delta)
        new_value = self.current_value + delta
        
        # Respect bounds
        self.current_value = max(self.min_bound, min(self.max_bound, new_value))

    def to_dict(self) -> dict:
        """Serialize parameter to a dictionary."""
        return {
            "name": self.name,
            "current_value": self.current_value,
            "min_bound": self.min_bound,
            "max_bound": self.max_bound,
            "drift_range": self.drift_range,
            "is_locked": self.is_locked
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Reconstruct parameter from a dictionary."""
        return cls(**data)

    def __repr__(self) -> str:
        status = "[LOCKED]" if self.is_locked else ""
        return f"{self.name}: {self.current_value:7.2f} {status}"
