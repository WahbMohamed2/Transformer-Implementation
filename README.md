# Notebook Organization Guide

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
