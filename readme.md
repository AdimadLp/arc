# ARC-Benchmarking Tutorial

## Description

This is the code of the Medium post [Exploring AI’s Complexities: Unraveling LLMs Abstract Thinking with ARC](https://medium.com/@adimadlp/exploring-ais-complexities-unraveling-llms-abstract-thinking-with-arc-20f94826207ci)
## Table of Contents

- [Installation](#installation)
    - [Installation on your own Hardware (linux)](#on-your-own-hardware-linux)
    - [Installation on your own Hardware (windows)](#on-your-own-hardware-windows)
- [Usage](#usage)
    - [Use on your own Hardware](#on-your-own-hardware)
    - [Use on Google Colab](#on-google-colab)

## Installation
### On your own Hardware (linux)

- [NVIDEA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

```bash
python3 pip install torch numpy transformers matplotlib seaborn
```

### On your own Hardware (windows)
- Follow this [Tutorial](https://thesecmaster.com/step-by-step-guide-to-setup-pytorch-for-your-gpu-on-windows-10-11/)

## Usage

### On your own Hardware
Here are examples of how to use this project:

```bash
# Start the traing
python3 train_gpt2.py
```
```bash
# Continue the training from checkpoint
python3 train_gpt2.py -c CHECKPOINT_FOLDERNAME
```
(e.g. CHECKPOINT_FOLDERNAME = gpt2_5e-05)

### On Google Colab
- Open the prepared [Notebook](colab_training.ipynb) in Colab and choose GPU Hardware accelerator
- Mount to your Google Drive to save checkpoints
- Copy your checkpoint destination folder starting with "/content/drive/MyDrive"
- Paste it in 
```python
train.train(model_path, learning_rate, batch_size, google_drive_path="/content/drive/MyDrive")
```