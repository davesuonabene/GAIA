# GenMix (Project Gaia)

GenMix is an evolutionary audio processing system that uses genetic algorithms to evolve multi-band mixing chains. It allows users to "breed" the perfect mix by selecting parent configurations and applying structural and parametric mutations.

## Core Architecture

### 1. Evolutionary DNA (`core/`)
- **`Parameter`**: The base unit of evolution. Holds value, bounds, and drift ranges. Supports locking to prevent further mutation.
- **`AudioModule`**: Base class for DSP effects. Currently includes a `CompressorModule` with safe-spawning defaults.
- **`Band`**: Manages a chain of `AudioModule`s for a specific frequency range. Handles structural mutations (adding/removing/swapping FX).
- **`Mix`**: The top-level genotype. Manages multiple `Band` objects separated by crossover frequencies.

### 2. Audio Engine (`audio/`)
- **`Crossover`**: A phase-accurate, mathematically transparent multi-band splitter. Uses a subtractive filter bank approach to ensure `Sum(Bands) == Input` with machine precision (`1e-16`).
- **`Engine`**: Maps GenMix DNA to real-time DSP using the `pedalboard` library.
- **`Analyzer`**: Provides fitness feedback using `librosa`. Extracts RMS (loudness), Transient Density (punch), and Spectral Centroid (brightness).

### 3. Genetic Algorithm (`ga/`)
- **`Population`**: Manages a batch of `Mix` candidates. Handles cloning and the generation of new offspring from a selected parent.

### 4. Interactive Interface (`cli.py`)
A keyboard-driven TUI (Terminal User Interface) built with `rich` and `InquirerPy`.
- View the current population in a color-coded table.
- Select a "Parent" for the next generation.
- Lock specific parameters to preserve "sweet spots."
- Adjust mutation rates for structure and parameters.
- Evolve and watch the DSP chains transform in real-time.

## Installation

```bash
# Navigate to the project directory
cd .openclaw/workspace/projects/gaia

# Activate the virtual environment
source venv/bin/bin/activate

# Install dependencies (if not already handled by venv setup)
pip install numpy scipy pedalboard librosa rich InquirerPy
```

## Usage

Launch the interactive evolution environment:
```bash
PYTHONPATH=. python cli.py
```

### Controls
- **Select Parent**: Pick the mix configuration you like most.
- **Lock Parameter**: Deep-dive into a module to freeze specific knobs.
- **Evolve Next Gen**: Generate a new batch of 5 children based on your selection.

## Testing & Verification
Each module includes a standalone test block to verify its logic:
- `core/parameter.py`: Test value drifting and locking.
- `audio/crossover.py`: Verify mathematical transparency of the filter bank.
- `audio/analyzer.py`: Confirm accurate detection of brightness and transients.
