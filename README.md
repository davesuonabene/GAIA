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
│   └── crossover.py       # Band-splitting stateful filter logic
├── STEM-CACHE/            # Persistent storage for separated stems
...
```

## 🎮 Interactive TUI (cli.py)

The interface uses a high-density, responsive layout designed for professional audio workflows.

### UI Features
*   **Contextual Header**: Displays the current file name and evolutionary generation.
*   **Stems Menu**: Dedicated menu for triggered high-fidelity source separation.
*   **Miller Column Bands**: Each frequency band is rendered as a discrete panel.
    *   **Frequency Title**: The crossover frequency is pinned to the **top-right** of each band.
    *   **Mixing Subtitle**: Band-level Gain is centered at the **bottom** of the panel.
*   **Pre-Mix Stems Row**: Dedicated panel for adjusting source balance before DSP.
*   **Focus Highlighting**: Active zones (Menu, Population, Stems, Bands, Playback) are visually distinguished by color.

### Navigation Controls
*   **TAB**: Cycle focus globally between the `MENU`, `POPULATION`, `STEMS`, `BANDS`, and `PLAYBACK` zones.
*   **Arrows (↑ / ↓ / ← / →)**: Navigate contextually *within* the currently focused zone.
*   **CTRL + Arrows**: Rapid navigation and 5x granular precision for value adjustments.
*   **K**: Toggle **Lock** status for the currently selected parameter.
*   **M / S**: 
    *   In **Bands Row**: Toggle **Mute** or **Solo** for the selected frequency band.
    *   In **Stems Row**: Toggle **Mute** or **Solo** for the selected source stem (Vocals, Drums, Bass, Other).
*   **Brackets ([ / ])**: Quick-switch between frequency Bands globally.
*   **SPACE / P**: Toggle **Playback** 🔊 of the current mix.
*   **ENTER**: 
    *   In **Population Row**: Evolve a new generation from the selected parent.
    *   In **Bands Row**: Enter **Inline Edit** mode or add missing effects.
*   **L**: Return to **File Browser**.
*   **E**: **Export** the current mix to a WAV file.

## 🧬 Stem Separation & Caching

GAIA features integrated source separation powered by the **Demucs (HTDemucs)** model.

*   **Background Processing**: Stem separation runs in a dedicated thread with a real-time progress bar in the TUI.
*   **STEM-CACHE**: Separated stems are saved permanently in the `STEM-CACHE/` directory.
*   **Auto-Load**: GAIA automatically detects if a file has been separated before and loads cached stems instantly upon track selection.
*   **Pre-Mixer**: Adjust the volume, mute, or solo source stems (Vocals, Drums, Bass, Other) before they enter the evolutionary effects chain.

## 🔬 Analysis Engine
The `audio/analyzer.py` system calculates feature data for evolutionary fitness.
*   **Configurable Pipeline**: Extracts RMS, transients, spectral centroid, BPM, and MFCC coefficients.

## 🚀 Getting Started

1.  **Environment Setup**:
    ```bash
    cd .openclaw/workspace/projects/gaia
    source venv/bin/activate
    # Ensure demucs and torch dependencies are installed
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
