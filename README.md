# Transformer Implementation

A modular, from-scratch implementation of the Transformer architecture in PyTorch, designed for educational purposes and research experimentation.

## Overview

This repository provides a comprehensive, step-by-step walkthrough of the Transformer architecture introduced in "Attention Is All You Need" (Vaswani et al., 2017). Each core component is implemented independently with accompanying Jupyter notebooks that illustrate the theory and practice behind modern attention-based models.

## Features

- **Modular Architecture**: Clean, decoupled implementation of Transformer components
- **Educational Notebooks**: Interactive tutorials for each architectural component
- **PyTorch Native**: Built entirely with PyTorch for seamless integration and extension
- **Research-Ready**: Designed for easy modification and experimentation
- **Well-Documented**: Clear explanations and references throughout

## Project Structure

```
transformer-implementation/
│
├── docs/                           # Component-level documentation
│   └── components.md
│
├── examples/                       # Training templates and usage examples
│   └── train_template.py
│
├── notebooks/                      # Educational Jupyter notebooks
│   ├── SelfAttention (1).ipynb
│   ├── multiHeadAttention (1).ipynb
│   ├── PositionalEncoding (1).ipynb
│   ├── layerNormalization.ipynb
│   ├── EncoderBlock.ipynb
│   └── DecoderBlock.ipynb
│
├── src/                            # Core implementation modules
│
├── tests/                          # Unit tests
│
├── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Jupyter Notebook or JupyterLab

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/transformer-implementation.git
cd transformer-implementation
pip install -r requirements.txt
```

## Learning Path

The notebooks are designed to be followed sequentially, with each building upon concepts from the previous:

1. **Self-Attention** - Core attention mechanism and scaled dot-product attention
2. **Multi-Head Attention** - Parallel attention heads for richer representations
3. **Positional Encoding** - Injecting sequence order information
4. **Layer Normalization** - Stabilizing training dynamics
5. **Encoder Block** - Complete encoder layer with attention and feed-forward networks
6. **Decoder Block** - Decoder layer with masked self-attention and cross-attention

Each notebook includes:
- Theoretical background
- Step-by-step implementation
- Visualization of intermediate outputs
- Practical examples

## Usage

### Running Notebooks

Launch Jupyter and navigate to the `notebooks/` directory:

```bash
jupyter notebook
```

Start with `SelfAttention (1).ipynb` and proceed through the sequence outlined above.

### Training Example

Refer to `examples/train_template.py` for a template on how to instantiate and train a complete Transformer model.

## Documentation

Detailed component documentation is available in `docs/components.md`, including:
- Architectural decisions
- Parameter specifications
- Mathematical formulations
- Implementation notes

## Testing

Run unit tests to verify component implementations:

```bash
pytest tests/
```

## References

- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - Harvard NLP
- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for:
- Bug fixes
- Documentation improvements
- Additional examples or tutorials
- Performance optimizations

## License

This project is available under the MIT License. See LICENSE file for details.

## Author

**Wahb Mohamed**

Developed as an educational resource for understanding deep learning and natural language processing architectures.

## Acknowledgments

This implementation is inspired by the seminal work of Vaswani et al. and serves as a learning tool for those interested in understanding Transformer models from first principles.

---

*For questions or feedback, please open an issue on GitHub.*
