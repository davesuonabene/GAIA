# Gaia GenMix: Evolutionary Audio Mixing System

Gaia GenMix is an interactive, hierarchical genetic algorithm (IGA) designed for audio mixing. It allows users to evolve multi-band effects chains through a specialized Terminal User Interface (TUI). Instead of static presets, GenMix treats every mixing decision as a gene that can be drifted, locked, or mutated.

## 🏗 Hierarchical Architecture

The system is built on a 4-tier hierarchy:

1.  **Level 1: Mix (The Organism)**
    *   Manages the global crossover frequencies and a collection of frequency bands.
    *   DNA: `crossover_params` (sorted frequency points).
2.  **Level 2: Band (The Chromosome)**
    *   A specific frequency range (e.g., Low, Mid, High) containing a sequential effect chain.
    *   Evolution: Structural mutations (add, remove, or swap effects).
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
│   ├── band.py           # Level 2: FX chain management & structural evolution
│   └── mix.py            # Level 1: Global crossover and band coordination
├── ga/                    # Genetic Algorithm Logic
│   └── population.py      # Population management and next-gen spawning
├── audio/                 # Audio Engine (DSP & Analysis)
│   ├── analyzer.py        # Feature extraction (RMS, Transients, etc.)
3.  **Level 3: AudioModule (The Gene Group)**
    *   DSP processors: `Compressor`, `Expander`, `Transient Shaper`, and `Saturation`.
    *   DNA: A set of internal `Parameter` objects.

...

## 🎮 Interactive TUI (cli.py)

The interface uses a responsive layout for deep-diving into the population's DNA and real-time playback control, built with a Focus Zone state machine.

### Navigation Controls
*   **TAB**: Cycle focus globally between the `MENU`, `POPULATION`, `BANDS` (horizontally), and `PLAYBACK` zones.
*   **Arrows (↑ / ↓ / ← / →)**: Navigate contextually *within* the currently focused zone.
    *   *In Population*: Select different mix candidates.
    *   *In Bands*: Navigate modules and adjust parameters.
    *   *In Playback*: Scrub audio backward and forward by 5 seconds.
*   **Brackets ([ / ])**: Switch between different frequency Bands.
*   **SPACE / P**: Toggle **Playback** 🔊 of the current mix.
*   **ENTER**: 
    *   In **Population Row**: Evolve a new generation from the selected parent.
    *   In **Bands Row**: Enter **Inline Edit** mode for the selected parameter.
*   **Plus (+) / Minus (-)**: Manually nudge a parameter value.
*   **L**: Return to **File Browser** to load a different audio file.
*   **E**: **Export** the current mix to a WAV file.
*   **ESC**: Cancel inline editing.
*   **CTRL+C**: Exit the application.

### Key Features
*   **Audio Engine**: Powered by `pedalboard` and `sounddevice`. Features a master Limiter to prevent clipping.
*   **Saturation Module**: New DSP module with `Drive` and `Mix` controls for harmonic enrichment.
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
