# GPT Model Implementation from Scratch using PyTorch

## Overview

This project is an implementation of the core architecture of the GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. GPT is a transformer-based model designed to generate human-like text based on input prompts. It has been widely used for various NLP tasks, such as text completion, translation, and summarization.

The implementation in this repository covers the key components of the GPT architecture, including the multi-head self-attention mechanism, positional encodings, and the transformer decoder blocks. The goal of this project is to provide a clear and concise implementation that helps in understanding the underlying mechanics of the GPT model.

## Features

- **Transformer Decoder Blocks**: The core component of the GPT model, consisting of multi-head self-attention, layer normalization, and feed-forward networks.
- **Positional Encodings**: Adds position information to the input embeddings, enabling the model to capture the order of words in a sequence.
- **Multi-Head Self-Attention**: Allows the model to focus on different parts of the input sequence simultaneously, capturing various dependencies.
- **Layer Normalization**: Ensures that the inputs to the neural network layers are well-conditioned, leading to more stable training.
- **Token Embeddings**: Converts tokens into dense vectors that the model can process.
- **Training and Inference**: The implementation supports both training from scratch and generating text using a pre-trained or trained model.

## Directory Structure

```
├───artifacts
│   ├───checkpoints
│   └───files
├───data
│   ├───processed
│   └───raw
├───logs
├───notebooks
├───src
└───unittest
```

## Requirements

To run this project, you need to have the following installed:

- Python 3.7+
- PyTorch 1.8+
- NumPy
- tqdm (for progress bars)
- Any other dependencies can be installed using the `requirements.txt` file.

You can install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

To train the GPT model from scratch, you can use the `train.py` script. This script will load the dataset, tokenize the text, and train the model.

```bash
python train.py --data_path data/dataset.txt --epochs 10 --batch_size 32 --learning_rate 0.001
```


### 3. Customizing the Model

You can modify the model architecture, training parameters, and other components by editing the respective files. The core model is implemented in `gpt_model.py`, where you can tweak the number of layers, heads, and other hyperparameters.

## Acknowledgments

This implementation is inspired by the original GPT paper by OpenAI and various open-source implementations. It is designed for educational purposes to help understand the core concepts of the GPT model.

## License

This project is licensed under the MIT License. Feel free to use, modify, and distribute this code as you wish.
