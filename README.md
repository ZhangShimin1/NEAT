# AcouSpike

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> A modern, lightweight library for Neuromorphic Audio Processing using Spiking Neural Networks (SNNs).

## 🌟 Overview

AcouSpike is a PyTorch-based framework designed for neuromorphic audio processing using Spiking Neural Networks (SNNs). It provides a flexible and efficient way to build, train, and deploy SNN models for various audio processing tasks.

## 🚀 Features

### 🧠 Rich Neuron Models (Already Supported)
AcouSpike implements a wide variety of spiking neuron models, ranging from classic to state-of-the-art:
- [x] **Classic:** LIF (Leaky Integrate-and-Fire), PLIF (Parametric LIF)
- [x] **Advanced:** ALIF (Adaptive LIF), GLIF (Gated LIF), RLIF (Recurrent LIF)
- [x] **Specialized:** CLIF, CELIF, TCLIF, adLIF, LTC, PMSN, DHSNN, SPSN
- [x] **Surrogate Gradients:** Built-in support for various surrogate gradient methods for direct training.

### 🏗️ Network Architectures (Already Supported)
Easily build and experiment with modern SNN architectures:
- [x] **Spikeformer:** Spiking Transformer networks
- [x] **Spiking CNN:** Spiking ResNet and other convolutional backbones
- [x] **Recurrent:** Spiking LSTM, Recurrent LIF
- [x] **Sequential:** TCN (Temporal Convolutional Networks), SSM (State Space Models)

### 🎧 Supported Audio Tasks (Recipes, Already Supported)
Ready-to-use recipes and training scripts for common audio applications:
- [x] **Automatic Speech Recognition (ASR):** End-to-end SNN-based speech recognition.
- [x] **Keyword Spotting (KWS):** Low-power keyword detection (e.g., Google Speech Commands).
- [x] **Speaker Identification:** Classifying speaker identities (e.g., VoxCeleb).
- [x] **Speaker Verification:** Verifying claimed speaker identities.
- [x] **Auditory Attention Decoding (AAD):** Decoding attended speech sources from neural signals.

## 📂 Project Structure

```text
AcouSpike/
├── acouspike/              # Core library
│   ├── models/
│   │   ├── neuron/         # Neuron implementations (LIF, PLIF, etc.)
│   │   ├── network/        # Network backbones (Spikeformer, TCN, etc.)
│   │   └── surrogate/      # Surrogate gradient functions
│   └── src/                # Training utilities, optimization, logging
├── recipes/                # Task-specific training scripts
│   ├── asr/
│   ├── keyword_spotting/
│   ├── speaker_identification/
│   ├── speaker_verification/
│   └── auditory_attention_decoding/
└── utils/                  # General utility functions
```

## 🔧 Installation

This project uses **[uv](https://github.com/astral-sh/uv)** for ultra-fast dependency management and packaging.

### 1. Install `uv`
```bash
# On Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Set up the environment
Clone the repo and sync dependencies.
```bash
git clone https://github.com/ZhangShimin1/AcouSpike
cd AcouSpike

# Creates a .venv folder and installs dependencies from pyproject.toml
uv sync
```

### 3. Activate the environment
```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

## ⚡ Quick Start

Each task in the `recipes/` directory comes with its own `run.sh` or instruction set.

**Example: Speaker Identification**

```bash
cd recipes/speaker_identification

# Run the training script (ensure your environment is activated)
bash run.sh
```

**Example: Keyword Spotting**

```bash
cd recipes/keyword_spotting

# Check the configuration files in conf/ and run
bash run.sh
```

## 📚 Documentation & Recipes

Detailed documentation for specific tasks can be found in their respective directories:

- [Automatic Speech Recognition (ASR)](./recipes/asr/README.md)
- [Keyword Spotting](./recipes/keyword_spotting/README.md)
- [Speaker Identification](./recipes/speaker_identification/README.md)
- [Speaker Verification](./recipes/speaker_verification/README.md)
- [Auditory Attention Decoding](./recipes/auditory_attention_decoding/readme.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on setting up your development environment, coding standards, and submission process.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

- Issue Tracker: [GitHub Issues](https://github.com/ZhangShimin1/AcouSpike/issues)
