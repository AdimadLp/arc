# ARC-Benchmarking Tutorial

## Description

This is the code of the Medium post [Exploring AI’s Complexities: Unraveling LLMs Abstract Thinking with ARC](https://medium.com/@adimadlp/exploring-ais-complexities-unraveling-llms-abstract-thinking-with-arc-20f94826207ci)
## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation
### On your own hardware (linux)

- [NVIDEA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

```bash
python3 pip install torch numpy transformers matplotlib seaborn
```

### On your own hardware (windows)
- See this [tutorial](https://thesecmaster.com/step-by-step-guide-to-setup-pytorch-for-your-gpu-on-windows-10-11/)

## Usage

Here are examples of how to use this project:

```bash
# Start the traing
python3 train_gpt2.py
```
```bash
# Continue the training from checkpoint
python3 train_gpt2.py -c FOLDERNAME
```