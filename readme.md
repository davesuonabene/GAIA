🧬 GenMix: Hierarchical Genetic Automixer
GenMix is an offline, non-linear, AI-assisted audio mixing environment. It uses an Interactive Genetic Algorithm (IGA) to evolve multi-band effects chains. Instead of a "black box" automixer, GenMix acts as an inspiration engine: it analyzes the audio, generates mutated variations of FX chains, and relies on the user to guide the evolution by rating mixes, locking parameters, and controlling mutation drift.

🏗 Core Philosophy
Minimal & Modular: No bloat. Every effect, band, and parameter is a self-contained evolutionary agent.

High Chaos Control: Strict separation between Structural Evolution (adding/removing FX) and Parametric Evolution (turning knobs).

Safe-Spawning: When the GA structurally spawns a new effect, it initializes in an acoustically transparent state (e.g., 1:1 compression, 0% drive) to prevent audio blowout.

UI Agnostic: Core logic is completely decoupled from the UI. Phase 1 uses a rich CLI. Phase 2 drops in gradio for instant web-based audio auditioning.

📐 System Architecture (The 4 Levels)
The software is built on a 4-tier Hierarchical Genetic Algorithm (HGA).

Level 1: Mix (Global)
Role: Manages the entire audio file and the multiband splitting logic.

DNA: Crossover frequencies (e.g., [150Hz, 2500Hz]) and a list of Band objects.

Evolution: Shifts crossover points, adds/removes frequency bands.

Level 2: Band (FX Chain)
Role: A specific frequency range containing an ordered FX chain.

DNA: A list of AudioModule objects (e.g., [EQ -> Compressor -> Saturation]).

Evolution (Structural): Inserts, deletes, or reorders FX modules.

Level 3: AudioModule (FX Unit)
Role: The DSP processor (wrapping pedalboard effects).

DNA: A dictionary of Parameter objects.

Evolution (Parametric): Mutates internal parameters. Contains the process_audio() method.

Level 4: Parameter (The Gene)
Role: The atomic building block.

DNA: name, current_value, min_bound, max_bound, drift_range (%), is_locked (Boolean).

Evolution: Mutates its value based on hill-climbing logic and user-defined drift.

🛠 Tech Stack
Audio DSP: pedalboard (Spotify's VST/Effects library), scipy.signal (Linkwitz-Riley crossovers).

Audio Analysis: librosa (transients, RMS, spectral centroid).

CLI Interface (Phase 1): rich (terminal formatting), InquirerPy (interactive menus), sounddevice (playback).

Web UI (Phase 2): gradio (auto-generated UI with native audio players).

📂 Directory Structure
Plaintext
genmix/
│
├── core/                   # The Evolutionary DNA
│   ├── parameter.py        # Level 4
│   ├── audio_module.py     # Level 3
│   ├── band.py             # Level 2
│   └── mix.py              # Level 1
│
├── audio/                  # DSP & Analysis
│   ├── crossover.py        # Linkwitz-Riley splitting
│   ├── analyzer.py         # Librosa feature extraction
│   └── engine.py           # Pedalboard rendering 
│
├── ga/                     # Evolutionary Logic
│   ├── population.py       # Manages generations and batch processing
│   └── mutations.py        # Math for parametric drift and structural spawning
│
├── cli.py                  # Phase 1: Rich + InquirerPy interface
├── web_ui.py               # Phase 2: Gradio interface (Future)
└── requirements.txt
