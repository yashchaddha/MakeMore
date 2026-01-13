# MakeMore - Character-Level Language Model

A character-level language model implementation for generating names, inspired by [Andrej Karpathy's Neural Network Training](https://github.com/karpathy/nn-zero-to-hero). This project demonstrates the fundamentals of neural network training from scratch, implementing both a simple bigram counting model and a neural network-based approach.

## Overview

MakeMore is an educational project that teaches the core concepts of:
- Character-level language modeling
- Bigram probability estimation
- Neural network implementation from scratch
- Gradient descent and backpropagation
- Text generation using trained models

## Features

- **Bigram Counting Model**: A simple frequency-based approach to character prediction
- **Neural Network Model**: A single-layer neural network trained with gradient descent
- **Name Generation**: Generate new names based on learned character patterns
- **Visualization**: Bigram frequency heatmaps for data exploration
- **Training Metrics**: Track loss and model performance during training

## Project Structure

```
MakeMore/
├── makemore.ipynb    # Main Jupyter notebook with implementation
├── names.txt         # Training dataset (list of names)
└── README.md         # This file
```

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install torch numpy matplotlib
```

## Usage

### Running the Notebook

1. Open `makemore.ipynb` in Jupyter Notebook or JupyterLab
2. Run the cells sequentially to:
   - Load and explore the names dataset
   - Build a bigram counting model
   - Train a neural network model
   - Generate new names

### Key Components

#### 1. Data Preparation
- Loads names from `names.txt`
- Creates character-to-index mappings (stoi/itos)
- Prepares training data as bigram pairs

#### 2. Bigram Counting Model
- Counts character pair frequencies
- Computes probability distributions
- Generates names using multinomial sampling

#### 3. Neural Network Model
- Single-layer network with 27x27 weight matrix
- One-hot encoding for input characters
- Softmax activation for probability distribution
- Trained using gradient descent with negative log-likelihood loss

#### 4. Name Generation
- Samples characters sequentially based on learned probabilities
- Stops when end-of-sequence token is generated
- Supports reproducible generation via random seed

## How It Works

1. **Character Encoding**: Each character (including special tokens for start/end) is mapped to an index (0-26)
2. **Training**: The model learns to predict the next character given the current character
3. **Generation**: Starting from a special start token, the model samples the next character based on learned probabilities, continuing until an end token is generated

## Example Output

After training, the model can generate names like:
- `cexzmazjallurailezkaynnellzimittain.`
- `llaynzkanza.`
- `stazthubrtthrigotai.`
- `mozjellavo.`

(Note: The quality improves with better training and more sophisticated architectures)

## Learning Objectives

This project covers:
- ✅ Character-level language modeling
- ✅ Probability distributions and sampling
- ✅ Neural network forward pass
- ✅ Loss functions (negative log-likelihood)
- ✅ Backpropagation and gradient descent
- ✅ One-hot encoding
- ✅ Softmax and probability normalization

## Future Enhancements

Potential improvements:
- Multi-layer neural networks
- Transformer architecture
- Better training techniques (learning rate scheduling, etc.)
- Evaluation metrics (perplexity, etc.)
- Model checkpointing and saving

## References

- Inspired by [Andrej Karpathy's Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)
- Based on the "makemore" series of tutorials

## License

This is an educational project. Feel free to use and modify for learning purposes.

