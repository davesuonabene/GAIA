# GAIA: Evolutionary Audio Mixing System

GAIA is an interactive, hierarchical genetic algorithm (IGA) designed for audio mixing. It allows users to evolve multi-band effects chains through a specialized Terminal User Interface (TUI). Instead of static presets, GAIA treats every mixing decision as a gene that can be drifted, locked, or mutated.

## 🏗 Hierarchical Architecture

The system is built on a 4-tier hierarchy:

1.  **Level 1: Mix (The Organism)**
    *   Manages global crossover frequencies, frequency bands, and global **PRE** and **POST** processing bands.
    *   DNA: `crossover_params`, `pre_band`, and `post_band`.
2.  **Level 2: Band (The Chromosome)**
    *   A processing unit (either a frequency range or a global PRE/POST layer) containing a sequential effect chain.
    *   **Constraints**: Each band can hold a maximum of **ONE** instance of each module type (e.g., 1 Compressor, 1 Saturation, etc.).
    *   Features: Mute, Solo (frequency bands only), and an evolvable **Gain** parameter.
    *   Evolution: Structural mutations and parameter drift.
3.  **Level 3: AudioModule (The Gene Group)**
    *   A DSP processor (Compressor, Expander, Transient Shaper).
    *   DNA: A set of internal `Parameter` objects.
4.  **Level 4: Parameter (The Gene)**
    *   The atomic unit: Threshold, Ratio, Gain, etc.
    *   DNA: Value, bounds, drift range, and a `is_locked` status.

## 📁 Project Structure

```text
gaia/
├── core/                  # Core Evolutionary Objects
│   ├── parameter.py       # Level 4: Value bounds and mutation logic
│   ├── audio_module.py    # Level 3: DSP units with safe-spawning
│   ├── band.py           # Level 2: FX chain management & mixing controls
│   └── mix.py            # Level 1: Global crossover and band coordination
├── ga/                    # Genetic Algorithm Logic
│   └── population.py      # Population management and next-gen spawning
├── audio/                 # Audio Engine (DSP & Analysis)
│   ├── analyzer.py        # Feature extraction (RMS, Transients, etc.)
│   ├── engine.py          # Routing and DSP execution
...
```

## 🎮 Interactive TUI (cli.py)

The interface uses a high-density, responsive layout designed for professional audio workflows.

### UI Features
*   **Contextual Header**: Displays the current file name and evolutionary generation at a glance.
*   **Miller Column Bands**: Each frequency band is rendered as a discrete panel.
    *   **Frequency Title**: The crossover frequency is pinned to the **top-right** of each band.
    *   **Mixing Subtitle**: Band-level Gain is centered at the **bottom** of the panel.
    *   **Pro Layout**: DSP module names are **UPPERCASE** and **left-justified**, while parameters are **right-justified** for maximum visual separation. No truncation is applied if space permits.
*   **Focus Highlighting**: Active zones (Menu, Population, Bands, Playback) are visually distinguished by border colors and brightness. The Population panel is now a clean box without redundant tags.

### Navigation Controls
*   **SHIFT + Up / Down**: In the **Population** zone, swap the position of the selected mix with the one above or below it.
*   **TAB**: Cycle focus globally between the `MENU`, `POPULATION`, `BANDS`, and `PLAYBACK` zones.
*   **Arrows (↑ / ↓ / ← / →)**: Navigate contextually *within* the currently focused zone.
*   **CTRL + Arrows**: Rapid navigation and comparison.
    *   *In Bands (Up/Down)*: Jump directly to the previous/next Module (Gene Group) or Band Gain.
    *   *In Bands (Left/Right)*: Switch between different frequency bands to compare parameters instantly.
    *   *Granular Adjustments*: When adjusting a numeric value, holding CTRL provides 5x more precision.
    *   *In Playback (Left/Right)*: Granular audio scrub (1 second instead of 5).
*   **K**: Toggle **Lock** status for the currently selected parameter (prevents mutation).
*   **M / S**: Toggle **Mute** or **Solo** for the currently selected band.
*   **Brackets ([ / ])**: Quick-switch between frequency Bands globally.
*   **SPACE / P**: Toggle **Playback** 🔊 of the current mix.
*   **ENTER**: 
    *   In **Population Row**: Evolve a new generation from the selected parent.
    *   In **Bands Row**: 
        *   If `[+] Add FX` is selected: Manually add the first available missing module to the band.
        *   Otherwise: Enter **Inline Edit** mode for the selected parameter.
*   **L**: Return to **File Browser** to load a different audio file.
*   **E**: **Export** the current mix to a WAV file.
*   **ESC**: Cancel inline editing.
*   **CTRL+C**: Exit the application.

### Key Features
*   **Audio Engine**: Powered by `pedalboard` and `sounddevice`. Supports real-time Mute/Solo/Gain routing.
*   **Saturation Module**: DSP module with `Drive` and `Mix` controls for harmonic enrichment.
*   **Real-time Streaming**: Audio is processed continuously in a background thread, preventing UI lockups.
*   **Safe-Spawning**: New modules are initialized with transparent settings to prevent audio surprises.

## 🔬 Analysis Engine
The `audio/analyzer.py` system calculates feature data for evolutionary fitness.
*   **Configurable Pipeline**: A dataclass `AnalysisConfig` controls which features (RMS, transients, spectral centroid, BPM, MFCC) are extracted.
*   **Track Metadata**: The `core/metadata.py` module manages basic file info to be fed into the analyzer.

## 🚀 Getting Started

1.  **Environment Setup**:
    ```bash
    cd .openclaw/workspace/projects/gaia
    source venv/bin/activate
    ```

2.  **Launch the System**:
    ```bash
    # Run the interactive TUI
    python cli.py
    ```

## 🧪 Testing
Each module in `core/` and `audio/` contains a `__main__` block for unit testing.
```bash
python core/mix.py
python core/parameter.py
```
