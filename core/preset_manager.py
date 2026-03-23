import json
import os
try:
    from .mix import Mix
except (ImportError, ValueError):
    from mix import Mix

class PresetManager:
    """Manager for saving and loading Mix presets in JSON format."""

    @staticmethod
    def save_preset(mix: Mix, filepath: str) -> None:
        """
        Saves the given Mix object to a JSON file.
        
        Args:
            mix: The Mix instance to save.
            filepath: Destination path for the JSON preset.
        """
        data = mix.to_dict()
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_preset(filepath: str) -> Mix:
        """
        Loads a Mix object from a JSON file.
        
        Args:
            filepath: Path to the JSON preset file.
            
        Returns:
            The reconstructed Mix object.
            
        Raises:
            FileNotFoundError: If the preset file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preset file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        return Mix.from_dict(data)

if __name__ == "__main__":
    # Simple test for serialization and saving
    from audio_module import CompressorModule
    
    # 1. Create a Mix
    test_mix = Mix(crossovers=[100.0, 1000.0])
    test_mix.bands[0].modules.append(CompressorModule())
    
    # 2. Save it
    preset_path = "test_preset.json"
    print(f"Saving test preset to {preset_path}...")
    PresetManager.save_preset(test_mix, preset_path)
    
    # 3. Load it back
    print(f"Loading preset back from {preset_path}...")
    loaded_mix = PresetManager.load_preset(preset_path)
    
    # 4. Verify
    print(f"Original Mix Crossovers: {test_mix.crossover_frequencies}")
    print(f"Loaded Mix Crossovers: {loaded_mix.crossover_frequencies}")
    print(f"Bands match: {len(test_mix.bands) == len(loaded_mix.bands)}")
    print(f"Module count in Band 0: {len(loaded_mix.bands[0].modules)}")
    
    # Cleanup
    if os.path.exists(preset_path):
        os.remove(preset_path)
    print("Test complete.")
