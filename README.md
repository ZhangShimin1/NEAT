# AcouSpike 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.0.1-green.svg)](https://github.com/username/repo)

> A modern, lightweight library for Neuromorphic Audio Processing

## 🚀 Features

- Feature 1: Build your own SNN model with PyTorch
- Feature 2: Support for multiple neuron types
- Feature 3: Support for multiple audio tasks
- Free from dependencies
- Fully tested

##Model overview
### Neuron model
We provide several neuron models in this library. Here's a detailed overview:

| Neuron Type | Path | Class | Reference |
|------------|------|-------|-----------|
| Leaky Integrate-and-Fire (LIF) | `/models/neuron/lif.py` | `LIF` | [Wu et al., 2019](https://arxiv.org/abs/1809.05793) |
| Recurrent Integrate-and-Fire (RLIF) | `/models/neuron/lif.py` | `RLIF` |  |
| Non-Spiking Integrate-and-Fire (RLIF) | `/models/neuron/lif.py` | `NonSpikingLIF` |  |
| Adaptive Stochastic Gradient Learning LIF (ASGL-LIF) | `/models/neuron/lif.py` | `ASGL_LIF` | |
| Parametric LIF (PLIF) | `/models/neuron/lif.py` | `PLIF` | [Fang et al., 2021](https://arxiv.org/abs/2007.05785) |
| Adaptive LIF (ALIF) | `/models/neuron/lif.py` | `ALIF` | [Yin et al., 2021](https://www.nature.com/articles/s42256-021-00397-w) |
| Generalized LIF (GLIF) | `/models/neuron/lif.py` | `GLIF` | [Yao et al., 2022](https://openreview.net/forum?id=UmFSx2c4ubT) |
| Conductive LIF (CLIF) | `/models/neuron/lif.py` | `CLIF` | [Huang et al., 2024](https://openreview.net/pdf?id=yY6N89IlHa) |
| Conductive Exponential LIF (CELIF) | `/models/neuron/lif.py` | `CELIF` | [Chen et al., 2023](https://arxiv.org/abs/2308.15150) |
| SPSN | `/models/neuron/spsn.py` | `SPSN` | |
| Liquid Time-Constant (LTC) | `/models/neuron/ltc.py` | `LTC` | [yin et al., 2023](https://github.com/byin-cwi/sFPTT/blob/main/fptt/fptt_mnist/snn_models_LIF4_save4.py) |
| Probabilistic Multi-Spike Neuron (PMSN) | `/models/neuron/pmsn.py` | `PMSN` | |
| Double-Headed Spiking Neural Network (DHSNN) | `/models/neuron/dhsnn.py` | `DHSNN` | [Zheng et al., 2024](https://www.nature.com/articles/s41467-023-44614-z) |
| Adaptive LIF (adLIF) | `/models/neuron/lif.py` | `adLIF` | [Bittar , 2022](https://github.com/idiap/sparch) |

## 📦 Installation 

```bash
pip install acouspike
```

## 🔨 Usage
run the following command to train a model:
```bash
cd recipes/Speaker_identification
python run.sh
```

