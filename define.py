#!/usr/bin/env python3
"""
Transformer Project Organizer
Organizes Jupyter notebooks and creates proper project structure
"""

import os
import shutil
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")


def create_project_structure():
    """Create organized project directory structure"""

    # Define project structure
    structure = {
        "notebooks": [
            "SelfAttention.ipynb",
            "multiHeadAttention.ipynb",
            "PositionalEncoding.ipynb",
            "layerNormalization.ipynb",
            "EncoderBlock.ipynb",
            "DecoderBlock.ipynb",
        ],
        "src": [],  # For future Python modules
        "tests": [],  # For unit tests
        "docs": [],  # For additional documentation
        "examples": [],  # For example scripts
    }

    # Create directories
    base_dir = Path("transformer-implementation")
    base_dir.mkdir(exist_ok=True)

    for folder in structure.keys():
        (base_dir / folder).mkdir(exist_ok=True)
        print(f"✓ Created directory: {folder}/")

    # Create requirements.txt
    requirements = """torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
jupyter>=1.0.0
"""

    with open(base_dir / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    print("✓ Created requirements.txt")

    # Create .gitignore
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# PyTorch
*.pth
*.pt

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
"""

    with open(base_dir / ".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore)
    print("✓ Created .gitignore")

    # Create notebook organization guide
    notebook_guide = """# Notebook Organization Guide

## Learning Order

Follow this sequence for optimal understanding:

1. **SelfAttention.ipynb** (30-45 min)
   - Fundamental attention mechanism
   - Scaling and masking concepts
   - Variance analysis

2. **PositionalEncoding.ipynb** (20-30 min)
   - Sinusoidal position encodings
   - Frequency patterns
   - Sequence position injection

3. **layerNormalization.ipynb** (15-20 min)
   - Normalization technique
   - Learnable parameters (gamma, beta)
   - Stabilization effects

4. **multiHeadAttention.ipynb** (45-60 min)
   - Multiple parallel attention heads
   - Tensor reshaping and permutations
   - Concatenation and projection

5. **EncoderBlock.ipynb** (30-45 min)
   - Complete encoder layer
   - Residual connections
   - Feed-forward networks

6. **DecoderBlock.ipynb** (45-60 min)
   - Masked self-attention
   - Cross-attention mechanism
   - Full decoder architecture

## Total Learning Time: ~3-4 hours

## Quick Reference

### Key Dimensions
- d_model: 512 (model dimension)
- num_heads: 8 (attention heads)
- d_k, d_v: 64 (per-head dimension)
- ffn_hidden: 2048 (FFN expansion)
- max_seq_len: 200 (maximum sequence)

### Important Shapes
```
Input: (batch=30, seq=200, d_model=512)
Q, K, V: (batch=30, heads=8, seq=200, d_k=64)
Attention: (batch=30, heads=8, seq=200, seq=200)
Output: (batch=30, seq=200, d_model=512)
```
"""

    with open(base_dir / "notebooks" / "README.md", "w", encoding="utf-8") as f:
        f.write(notebook_guide)
    print("✓ Created notebooks/README.md")

    # Create example training script template
    training_template = """#!/usr/bin/env python3
\"\"\"
Example Training Script Template
This is a template for training a complete Transformer model
\"\"\"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# TODO: Import your Encoder and Decoder classes
# from src.encoder import Encoder
# from src.decoder import Decoder


class Transformer(nn.Module):
    \"\"\"Complete Transformer model\"\"\"
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1,
                 max_seq_length=200):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # TODO: Add positional encoding
        # self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # TODO: Add encoder and decoder
        # self.encoder = Encoder(d_model, num_heads, dropout, d_ff, num_layers)
        # self.decoder = Decoder(d_model, d_ff, num_heads, dropout, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embed and encode source
        src_embedded = self.src_embedding(src) * (self.d_model ** 0.5)
        # TODO: Add positional encoding
        # src_embedded = self.pos_encoding(src_embedded)
        
        # Embed target
        tgt_embedded = self.tgt_embedding(tgt) * (self.d_model ** 0.5)
        # TODO: Add positional encoding
        # tgt_embedded = self.pos_encoding(tgt_embedded)
        
        # Encode and decode
        # TODO: encoder_output = self.encoder(src_embedded, src_mask)
        # TODO: decoder_output = self.decoder(encoder_output, tgt_embedded, tgt_mask)
        
        # Project to vocabulary
        # TODO: output = self.output_projection(decoder_output)
        
        # return output
        pass


def train_step(model, src, tgt, optimizer, criterion):
    \"\"\"Single training step\"\"\"
    model.train()
    optimizer.zero_grad()
    
    # TODO: Implement training step
    # output = model(src, tgt[:, :-1])  # Teacher forcing
    # loss = criterion(output.view(-1, vocab_size), tgt[:, 1:].view(-1))
    # loss.backward()
    # optimizer.step()
    
    # return loss.item()
    pass


def main():
    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    
    # Initialize model
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, 
                       num_heads, num_layers, d_ff, dropout)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is padding
    
    print("Model initialized. Ready for training!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
"""

    with open(base_dir / "examples" / "train_template.py", "w", encoding="utf-8") as f:
        f.write(training_template)
    print("✓ Created examples/train_template.py")

    # Create component summary
    component_summary = """# Component Summary

## Core Components Implemented

### 1. Scaled Dot-Product Attention
```python
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention
```

### 2. Multi-Head Attention
- Projects input to Q, K, V
- Splits into multiple heads
- Applies scaled dot-product attention per head
- Concatenates and projects output

### 3. Positional Encoding
- Sinusoidal position embeddings
- Fixed (not learned)
- Allows model to use sequence order

### 4. Position-wise Feed-Forward Network
- Two linear layers with ReLU activation
- Expansion: d_model -> d_ff -> d_model
- Applied to each position independently

### 5. Layer Normalization
- Normalizes across feature dimension
- Learnable scale (gamma) and shift (beta)
- Applied with residual connections

### 6. Encoder Layer
- Multi-head self-attention
- Feed-forward network
- Two residual connections with layer norm

### 7. Decoder Layer
- Masked multi-head self-attention
- Multi-head cross-attention
- Feed-forward network
- Three residual connections with layer norm

## Shape Transformations

```
Encoder Flow:
Input: (B, L, d_model)
  -> Self-Attention: (B, L, d_model)
  -> Add & Norm: (B, L, d_model)
  -> FFN: (B, L, d_model)
  -> Add & Norm: (B, L, d_model)

Decoder Flow:
Input: (B, L, d_model)
  -> Masked Self-Attention: (B, L, d_model)
  -> Add & Norm: (B, L, d_model)
  -> Cross-Attention: (B, L, d_model)
  -> Add & Norm: (B, L, d_model)
  -> FFN: (B, L, d_model)
  -> Add & Norm: (B, L, d_model)
```

## Configuration Standards

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_model | 512 | Model dimension |
| num_heads | 8 | Number of attention heads |
| d_k, d_v | 64 | Dimension per head |
| d_ff | 2048 | FFN hidden dimension |
| dropout | 0.1 | Dropout probability |
| num_layers | 6 | Number of encoder/decoder layers |
"""

    with open(base_dir / "docs" / "components.md", "w", encoding="utf-8") as f:
        f.write(component_summary)
    print("✓ Created docs/components.md")

    print("\n" + "=" * 60)
    print("✓ Project structure created successfully!")
    print("=" * 60)
    print(f"\nProject location: ./{base_dir}/")
    print("\nNext steps:")
    print("1. Move your notebooks to the notebooks/ directory")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Follow the learning path in notebooks/README.md")
    print("4. Start with SelfAttention.ipynb")

    return base_dir


if __name__ == "__main__":
    project_dir = create_project_structure()

    print("\n" + "=" * 60)
    print("Project structure:")
    print("=" * 60)

    for root, dirs, files in os.walk(project_dir):
        level = root.replace(str(project_dir), "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
