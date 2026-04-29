# TinyMathReason-1B Architecture Design

## 1. Exact Model Configuration
- **Architecture**: Llama-style Decoder-only Transformer
- **Total Parameters**: ~1.12B
- `hidden_dim`: 2048
- `num_layers`: 22
- `num_attention_heads`: 16
- `num_kv_heads`: 4 (Grouped Query Attention ratio of 4:1)
- `head_dim`: 128 (hidden_dim / num_attention_heads)
- `intermediate_dim`: 5632 (SwiGLU)
- `vocab_size`: 32000
- `max_seq_len`: 4096
- `rms_norm_eps`: 1e-5
- `rope_theta`: 10000.0
- `tie_word_embeddings`: False
- `precision`: bfloat16

## 2. Layer-by-Layer Parameter Count
Here is the exact calculation to prove the total is ~1.12B:

- **Embeddings (Untied)**: 32,000 × 2048 = 65,536,000
- **Attention (Per Layer)**:
  - Q_proj: 2048 × 2048 = 4,194,304
  - K_proj: 2048 × (2048 × 4/16) = 1,048,576
  - V_proj: 2048 × (2048 × 4/16) = 1,048,576
  - O_proj: 2048 × 2048 = 4,194,304
  - Total Attention = 10,485,760
- **MLP (Per Layer)**:
  - Gate_proj: 2048 × 5632 = 11,534,336
  - Up_proj: 2048 × 5632 = 11,534,336
  - Down_proj: 5632 × 2048 = 11,534,336
  - Total MLP = 34,603,008
- **LayerNorms (2 per layer)**: 2048 × 2 = 4096
- **Total Per Layer**: 10,485,760 + 34,603,008 + 4096 = 45,092,864
- **Total for 22 Layers**: 45,092,864 × 22 = 992,043,008
- **Final LayerNorm**: 2,048
- **LM Head (Untied)**: 32,000 × 2048 = 65,536,000

**TOTAL PARAMETERS**: 65,536,000 + 992,043,008 + 2048 + 65,536,000 = **1,123,117,056** (~1.12B)

## 3. Architecture Comparison Table

| Feature | TinyMathReason-1B | TinyLlama-1.1B | SmolLM2-1.7B | Qwen2.5-0.5B | Qwen2.5-1.5B | Llama-3.2-1B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Params** | 1.12B | 1.1B | 1.7B | 0.5B | 1.5B | 1.2B |
| **Layers** | 22 | 22 | 24 | 24 | 28 | 16 |
| **Hidden Dim** | 2048 | 2048 | 2048 | 896 | 1536 | 2048 |
| **Intermed Dim**| 5632 | 5632 | 8192 | 4864 | 8960 | 8192 |
| **Attention Heads**| 16 | 32 | 32 | 14 | 12 | 32 |
| **KV Heads** | 4 | 4 | 32 (MHA) | 2 | 2 | 8 |
| **Head Dim** | 128 | 64 | 64 | 64 | 128 | 64 |
| **Vocab Size** | 32000 | 32000 | 49152 | 151936 | 151936 | 128256 |
| **GQA** | Yes (4:1) | Yes (8:1) | No (MHA) | Yes (7:1) | Yes (6:1) | Yes (4:1) |
| **Tie Embeds**| No | No | No | True | True | True |
| **Max Context** | 4096 | 2048 | 8192 | 32768 | 32768 | 131072 |

## 4. Architecture Choice Justifications

- **Why GQA 4:1?** Grouped Query Attention reduces the memory bandwidth required during inference (KV cache size) while maintaining performance close to Multi-Head Attention. A 4:1 ratio (16 Q heads, 4 KV heads) is the sweet spot used by modern models like Llama 3 to balance speed and quality.
- **Why 22 layers and 2048 hidden dim?** This exactly matches the TinyLlama base configuration, which has been battle-tested to provide excellent capacity for a 1.1B model without being too deep (which can cause training instability or slow inference) or too wide.
- **Why 16 attention heads instead of 32?** We use a head dimension of 128 (2048 / 16) instead of 64. A larger head dimension (128) is standard in newer architectures (like Llama 3) as it allows the model to capture more complex relationships within a single attention head, leading to better reasoning capabilities which is crucial for a math model.
- **Why 5632 intermediate dim?** The SwiGLU activation function uses two matrices instead of one. To keep the parameter count similar to a standard MLP with a 4x expansion factor, we use an intermediate dimension of roughly `8/3 * hidden_dim` (8/3 * 2048 = 5461.33), rounded up to a multiple of 128 (5632) for hardware efficiency on TPUs.
- **Why vocab size 32000?** A smaller vocabulary is sufficient for English and Math, and it significantly reduces the embedding and LM head parameter count, leaving more capacity for the transformer layers (the "reasoning engine"). A 32k vocab is highly optimized for English and standard math notation, similar to Llama 2.
- **Why max sequence length 4096?** This is long enough to fit multiple math problems, chain-of-thought derivations, and few-shot prompts during SFT and DPO, while remaining efficient enough to train rapidly on TPU v4-32 within our compute budget.
