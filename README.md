# GPT-2 Implementation from Scratch

A complete implementation of GPT-2 (Generative Pre-trained Transformer 2) built from the ground up using PyTorch.
##  Project Overview

This project implements a fully functional GPT-2 model without relying on pre-built transformer libraries. Every component, from preprocessing pipeline to attention mechanism to the final language model, has been coded from scratch to understand the inner workings of large language models.

##  Key Features

- **Data Preprocessing Pipeline**: Comprehensive text processing and tokenization
- **Self-Attention \& Multi-Head Attention**: Custom implementation of attention mechanism with causal masking for autoregressive generation
- **Complete Transformer Architecture**: Full implementation of encoder-decoder attention mechanisms
- **Advanced Activation Functions**: Custom GELU activation function implementation
- **Layer Normalization**: From-scratch implementation for training stability
- **Instruction Fine-tuning**: Implementation of supervised fine-tuning techniques
- **Modular Design**: Clean, reusable code structure following software engineering best practices


##  Technical Implementation

### Core Components Implemented

#### 1. **Data Processing** ([data_preprocessing.ipynb](data_preprocessing.ipynb))

- **Text Tokenization**: Custom tokenization pipeline
- **Sequence Preparation**: Context window management
- **Batch Processing**: Efficient data loading and batching

#### 2. **Attention Mechanism** ([Attention_mechanism.ipynb](Attention_mechanism.ipynb))

- **Self-Attention**: Query, Key, Value matrix computations with scaled dot-product attention[^1]
- **Causal Attention**: Upper triangular masking for autoregressive text generation[^1]
- **Multi-Head Attention**: Parallel attention heads with concatenation and linear projection[^1]
- **Dropout Integration**: Regularization techniques to prevent overfitting[^1]

```python
# Key features implemented:
- Attention score computation: queries @ keys.T
- Scaled attention: scores / sqrt(d_k) 
- Causal masking with -inf for future tokens
- Multi-head parallel processing
- Context vector generation
```


#### 3. **Transformer Architecture** ([Transformer_arch.ipynb](Transformer_arch.ipynb))

- **Layer Normalization**: Custom implementation for training stability[^2]
- **GELU Activation**: Gaussian Error Linear Unit for improved gradients[^2]
- **Feed-Forward Networks**: Position-wise fully connected layers[^2]
- **Residual Connections**: Skip connections for gradient flow[^2]
- **TransformerBlock**: Complete transformer layer with attention and FFN[^2]


#### 4. **GPT-2 Model Architecture** ([GPT-2.ipynb](GPT-2.ipynb))

- **Token Embeddings**: Learnable vocabulary representations
- **Positional Embeddings**: Position-aware input encoding
- **Stacked Transformer Layers**: 12-layer transformer stack
- **Language Model Head**: Final linear layer for next-token prediction

#### 5. **Instruction Fine-tuning** ([instruction_finetuning.ipynb](instruction_finetuning.ipynb))

- **Supervised Fine-tuning**: Task-specific model adaptation[^3]
- **Training Loop**: Custom training implementation with loss computation[^3]
- **Model Configuration**: GPT-2 124M parameter configuration[^3]


##  Project Structure

```
GPT-2-from-Scratch/
├── Attention_mechanism.ipynb      # Core attention implementations
├── data_preprocessing.ipynb       # Data pipeline and tokenization
├── GPT-2.ipynb                   # Complete GPT-2 model
├── Transformer_arch.ipynb        # Transformer building blocks
├── instruction_finetuning.ipynb  # Fine-tuning implementation
├── the-verdict.txt               # Sample training data
└── README.md                     # Project documentation
```


## Technical Specifications

- **Framework**: PyTorch
- **Model Size**: GPT-2 124M parameters
- **Context Length**: 1024 tokens
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Transformer Layers**: 12
- **Vocabulary Size**: 50,257 tokens


## Key Technical Achievements

### Deep Learning Concepts Mastered

- **Attention Mechanisms**: Scaled dot-product attention, multi-head attention
- **Transformer Architecture**: Layer normalization, residual connections, feed-forward networks
- **Activation Functions**: GELU implementation with mathematical precision
- **Regularization**: Dropout, layer normalization for training stability
- **Optimization**: Gradient flow through deep networks

##  Implementation Highlights

### Advanced Mathematical Concepts

```python
# Scaled Dot-Product Attention
attn_scores = queries @ keys.transpose(2, 3)
attn_scores = attn_scores / (head_dim ** 0.5) 
attn_weights = torch.softmax(attn_scores, dim=-1)
context_vectors = attn_weights @ values
```


### Causal Masking for Autoregressive Generation

```python
# Preventing attention to future tokens
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
attn_scores.masked_fill_(mask.bool(), -torch.inf)
```


### Custom GELU Activation

```python
# Gaussian Error Linear Unit implementation
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
        (x + 0.044715 * torch.pow(x, 3))
    ))
```


## Skills Demonstrated

**Machine Learning \& Deep Learning**

- Transformer architecture understanding
- Attention mechanism implementation
- Neural network optimization
- Gradient computation and backpropagation

**Programming \& Software Development**

- PyTorch framework expertise
- Object-oriented programming
- Code modularity and reusability
- Performance optimization

**Mathematical Foundations**

- Linear algebra applications
- Probability and statistics
- Calculus in neural networks
- Information theory concepts


##  Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib jupyter
```


### Running the Code

1. **Data Preprocessing**: Start with `data_preprocessing.ipynb`
2. **Attention Mechanism**: Explore `Attention_mechanism.ipynb`
3. **Architecture**: Review `Transformer_arch.ipynb`
4. **Complete Model**: Run `GPT-2.ipynb`
5. **Fine-tuning**: Execute `instruction_finetuning.ipynb`

##  Educational Value

This project serves as a comprehensive learning resource for:

- Understanding transformer architecture from first principles
- Implementing attention mechanisms without black-box libraries
- Mastering PyTorch for custom model development
- Learning modern NLP techniques and best practices

## Contact

**Developer**: Vibhor Bhatia
**Email**: vibhor.1bhatia@gmail.com
