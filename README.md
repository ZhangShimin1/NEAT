# AcouSpike 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> A modern, lightweight library for Neuromorphic Audio Processing using Spiking Neural Networks.

## 🌟 Overview

AcouSpike is a PyTorch-based framework designed for neuromorphic audio processing using Spiking Neural Networks (SNNs). It provides a flexible and efficient way to build, train, and deploy SNN models for various audio processing tasks.

## 🚀 Features

- **Flexible Architecture**
  - Build custom SNN models using PyTorch
  - Support for various neuron types and synaptic connections
  - Modular design for easy extension

- **Audio Processing**
  - Built-in support for common audio tasks
  - Efficient spike encoding for audio signals

- **Developer Friendly**
  - Minimal dependencies
  - Comprehensive documentation
  - Easy-to-follow examples

## 🔧 Installation & Development

This project uses **[uv](https://github.com/astral-sh/uv)** for ultra-fast dependency management and packaging. If you are new to `uv`, think of it as a modern replacement for `pip`, `pip-tools`, and `virtualenv` all in one binary.

### 1. Install `uv`
First, install the tool itself:

```bash
# On Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Set up the environment
Clone the repo and sync dependencies. `uv` will automatically create a virtual environment for you.

```bash
git clone https://github.com/ZhangShimin1/AcouSpike
cd AcouSpike

# This creates a .venv folder and installs all dependencies defined in pyproject.toml
uv sync
```

### 3. Activate the environment

To run commands (like python scripts), you can

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Now you can use 'python' directly
```

### 4. Adding new dependencies
If you need to add a new library (e.g., `scikit-learn`):

```bash
uv add scikit-learn
```

This automatically updates `pyproject.toml` and the lock file.

## 🚀 Quick Start

Run the speaker identification task:

```bash
cd recipes/speaker_identification
# Ensure you are in the virtual environment or use 'uv run'
bash run.sh
```

## 📚 Documentation

### Model Components

- [Neuron Models](./acouspike/models/neuron/README.md)
- [Network Architectures](./acouspike/models/SNN/README.md)

### Tutorials

1. [Getting Started](./docs/tutorials/getting_started.md)
2. [Building Your First SNN](./docs/tutorials/first_snn.md)
3. [Audio Processing Basics](./docs/tutorials/audio_processing.md)

## 📊 Benchmarks

Performance benchmarks and comparisons are available in our [benchmarks page](./docs/benchmarks.md).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📬 Contact

- Issue Tracker: [GitHub Issues](https://github.com/username/acouspike/issues)
- Email: maintainer@example.com

## 🙏 Acknowledgments

- List of contributors
- Supporting organizations
- Related projects and inspirations

