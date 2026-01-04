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

## 🔧 Installation

Install uv:

```bash
# On Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install AcouSpike:

```bash
git clone https://github.com/ZhangShimin1/AcouSpike
cd AcouSpike

# The `uv sync` command create a virtual environment and install all dependencies
uv sync
```

Run the tasks, for example, speaker identification:

```bash
cd recipes/speaker_identification
bash run.sh
```

Go to task directory and follow the instructions in the `README.md` file.

## 📚 Documentation

### Model Components

- [Neuron Models](./acouspike/models/neuron/README.md)
- [Network Architectures](./acouspike/models/SNN/README.md)

### Tutorials

1. [Getting Started](./docs/tutorials/getting_started.md)
2. [Building Your First SNN](./docs/tutorials/first_snn.md)
3. [Audio Processing Basics](./docs/tutorials/audio_processing.md)

## 🎯 Examples

Ready-to-use examples are available in the `recipes` directory:

- Speaker Identification
```bash
cd recipes/speaker_identification
python run.sh
```

- Keyword Spotting
```bash
cd recipes/keyword_spotting
python run.sh
```

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

