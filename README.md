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
│   ├── crossover.py       # Phase-accurate Linkwitz-Riley splitting
│   └── engine.py          # Pedalboard-based rendering
└── cli.py                 # Miller Column TUI & Keyboard Handler
```

## 🎮 Interactive TUI (cli.py)

The interface uses a **Miller Column** navigation system for deep-diving into the population's DNA.

### Navigation Controls
*   **Arrows (↑ / ↓)**: Move the cursor up and down within the current column.
*   **Arrows (← / →)**: Navigate between hierarchy levels (Population → Mix Tree → Modules/XO → Params).
*   **SPACE**: Toggle **LOCK** 🔒 on the selected parameter. Locked parameters will not mutate.
*   **ENTER**:
    *   At the **Population** level (far left): Evolve a new generation from the selected parent.
    *   At the **Parameter** level (far right): Enter **Inline Edit** mode to type a specific value.
*   **Plus (+) / Minus (-)**: Manually nudge a parameter value up or down.
*   **ESC**: Cancel inline editing.
*   **CTRL+C**: Exit the application.

### Key Features
*   **Safe-Spawning**: New modules are initialized with transparent settings (e.g., 1:1 ratio) to prevent audio surprises.
*   **Cross-Platform Input**: Low-level keyboard listener optimized for Linux (unbuffered `os.read`) and Windows.
*   **Real-time Feedback**: Color-coded columns show the path from the population down to the specific parameter.

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
