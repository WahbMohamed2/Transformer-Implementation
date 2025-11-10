# Component Summary

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
