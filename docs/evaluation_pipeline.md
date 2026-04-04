# Evaluation Pipeline: From Raw Bytes to BPB

This document traces every step of how the model is evaluated, from raw data on disk to the final BPB score.

## 1. Data: How Tokens Get Into the Evaluation

### The Validation Shard

The validation data lives in `data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin`. It's a binary file containing **62,021,633 uint16 token IDs**, packed contiguously with no padding and no sequence boundaries.

**File format:**
- First 1024 bytes: 256-element int32 header (`header[0]` = magic `20240520`, `header[2]` = token count)
- Remaining bytes: uint16 token IDs, little-endian, back to back

These token IDs were produced offline by tokenizing FineWeb text with the `fineweb_1024_bpe.model` SentencePiece tokenizer (1024-token BPE vocabulary). The original document boundaries are **not preserved** — the token stream is one long concatenated sequence.

### Loading and Slicing

`load_validation_tokens()` reads the shard into a numpy int32 array and truncates to a multiple of `seq_len` (1024) plus 1:

```
raw tokens: [t0, t1, t2, ..., t62021632]
usable = ((62021633 - 1) // 1024) * 1024 = 62,021,632
kept:      [t0, t1, ..., t62021632]  (62,021,633 tokens)
```

The +1 is critical: we need one extra token because the model predicts the *next* token, so the last input token's target is the token after it.

### Forming Sequences

The token stream is sliced into non-overlapping sequences of length 1024 (the standard eval mode; sliding window is separate):

```
Sequence 0:  input = [t0,    t1,    ..., t1023]     target = [t1,    t2,    ..., t1024]
Sequence 1:  input = [t1024, t1025, ..., t2047]     target = [t1025, t1026, ..., t2048]
Sequence 2:  input = [t2048, t2049, ..., t3071]     target = [t2049, t2050, ..., t3072]
...
```

Each sequence is 1024 tokens. The input and target are offset by 1 — `target[i] = input[i+1]`. This is standard autoregressive next-token prediction.

**Important:** Since the original data has no document boundaries, a single 1024-token sequence may span **multiple original documents**. The model has no way to know when one document ends and another begins within a sequence. This means:
- The first few tokens of a new document within a sequence get no useful context (the preceding tokens are from a different document)
- The high loss at position 0 (4.34 CE) partly reflects this: the very first token of each sequence has zero context

### Batching

Sequences are grouped into batches. With `val_batch_size=65536` and `seq_len=1024`, each batch contains 64 sequences (65,536 tokens). The model processes all 64 sequences in parallel.

Total sequences from 62M tokens: ~60,567. With `val_max_tokens=1,048,576`, we evaluate 1,024 sequences (1M tokens).

## 2. The Model: What Happens Inside

### Step 1: Token Embedding

Each input token ID (an integer 0-1023) is looked up in `tok_emb`, a 1024 x 640 embedding matrix. This produces a 640-dimensional vector for each token position.

```
input_ids: shape (batch, 1024) of integers
tok_emb:   shape (1024, 640) lookup table
output:    shape (batch, 1024, 640) — one 640-dim vector per position
```

### Step 2: Bigram Hash Embedding (additive)

For each consecutive pair of tokens (t_{i-1}, t_i), a hash function maps them to one of 4096 buckets:

```
hash = (36313 * t_i) XOR (27191 * t_{i-1}) mod 4095
```

The hash indexes into a learned 4096 x 128 embedding table, which is projected to 640 dimensions and **added** to the token embedding. This gives the model immediate access to bigram statistics without attention.

Position 0 has no previous token, so it uses a default bucket.

### Step 3: RMSNorm + SmearGate

**RMSNorm:** Each 640-dim vector is normalized to have unit RMS (root mean square). No learned scale/bias — just normalization.

**SmearGate:** A learned per-dimension gate (sigmoid) blends each position's embedding with the *previous* position's embedding:

```
output[t] = (1 - gate) * x[t] + gate * x[t-1]
```

This gives the model a cheap 1-token lookback in the embedding space, before any attention. Position 0 blends with a zero vector.

After SmearGate, the result is saved as `x0` (the "base stream" used by residual mixing throughout the network).

### Step 4: Transformer Blocks (U-Net Structure)

The model has 10 blocks (for the multi-skip model) or 13 blocks (for the 322 model), organized as a U-Net with encoder and decoder halves.

**Each block does:**
1. **Residual mixing:** Blend the current hidden state `x` with the base stream `x0` using learned per-dimension weights: `x = mix[0] * x + mix[1] * x0`
2. **Attention:** RMSNorm -> Causal Self-Attention -> Scaled residual add
3. **MLP:** RMSNorm -> MLP (with LeakyReLU^2 activation) -> Scaled residual add

**Causal Self-Attention in detail:**

For each position, the model computes:
- **Query, Key, Value** via three separate linear projections from the 640-dim hidden state
- With 10 heads and 5 KV heads (GQA), each head has dimension 64
- Q and K are RMSNorm'd, then **Partial RoPE** is applied (16 of 64 dims get rotational position encoding, 48 pass through unchanged)
- Q is scaled by a learned `qk_gain` (initialized to 1.5)
- Scaled dot-product attention with **causal mask**: position i can only attend to positions 0..i. This means:
  - Position 0 sees only itself (no context)
  - Position 100 sees positions 0-100 (100 tokens of context)
  - Position 1023 sees the full 1024-token window
- XSA (eXtended Self-Attention): after standard attention, the self-value component is projected out to reduce attention sink artifacts

**The causal mask is why position matters so much for prediction difficulty.** Early positions have very little context; late positions have up to 1024 tokens.

**MLP in detail:**
- Linear projection from 640 to `640 * mult` dimensions (mult varies per layer: 3, 2, or 0.5)
- LeakyReLU with slope 0.5, then squared: `lrelu(x)^2`
- Linear projection back from `640 * mult` to 640

**U-Net skip connections:**
- In the 13L/322 model: 6 encoder layers, 7 decoder layers. Each decoder gets a skip from its symmetric encoder partner (LIFO order). The skip is a weighted residual add.
- In the 10L multi-skip: 6 encoder, 4 decoder. Each of the first 3 decoders gets **two** skips (one early encoder, one late encoder).

### Step 5: Final Normalization

After all blocks, the output hidden states are RMSNorm'd one final time. Output shape: (batch, 1024, 640).

## 3. From Hidden States to Predictions

### Logit Computation (Tied Embeddings)

The model uses **tied embeddings**: the same 1024 x 640 matrix used for input embedding is reused as the output projection. The hidden state at each position is multiplied by the transpose of the embedding matrix:

```
logits = hidden @ tok_emb.weight.T
shape: (batch, 1024, 640) @ (640, 1024) = (batch, 1024, 1024)
```

Each position now has 1024 logit values — one per vocabulary token. Higher logit = model thinks that token is more likely to come next.

### Logit Softcap

Before computing probabilities, logits are passed through a soft capping function:

```
logits = 30.0 * tanh(logits / 30.0)
```

This squashes extreme logits to the range (-30, +30), preventing the model from being arbitrarily confident. Without this, a single extreme logit could dominate the softmax.

### What the Model "Predicts"

At each position i, the model outputs a probability distribution over all 1024 tokens for what should come at position i+1. The "prediction" is this full distribution, not a single token. The top-1 prediction (argmax of logits) is what the model considers most likely.

## 4. Measuring the Error: Cross-Entropy Loss

### Per-Token Loss

For each position, we compute the cross-entropy between the model's predicted distribution and the actual next token:

```
loss[i] = -log(softmax(logits[i])[target[i]])
```

In words: take the model's logit vector at position i, convert to probabilities via softmax, look up the probability assigned to the **actual** next token, take the negative log.

- If the model assigns probability 1.0 to the correct token: loss = 0 (perfect prediction)
- If the model assigns probability 0.5: loss = 0.693 (ln(2))
- If the model assigns probability 0.01: loss = 4.605 (high surprise)
- If the model assigns probability 0.001: loss = 6.908 (very wrong)

The loss is in **nats** (natural log). To convert to bits: divide by ln(2).

### Aggregation: Mean Cross-Entropy

The loss for a batch is the **mean** over all token positions in all sequences:

```
batch_loss = sum(loss[i] for all i) / total_tokens
```

This gives equal weight to every token prediction, regardless of whether it's a common word or a rare byte-fallback token.

### What "Good" vs "Bad" Loss Looks Like

For this model at convergence (~2.27 CE):
- **Loss < 0.5** (~35% of tokens): The model is very confident and correct. Typically common continuations like `'s'` after a word, `'the'` after `'of'`, closing punctuation.
- **Loss 0.5-1.0** (~6% of tokens): Model is reasonably confident, correct token is in top-3 or so.
- **Loss 1.0-3.0** (~32% of tokens): Model has moderate uncertainty. The correct token is plausible but not dominant.
- **Loss 3.0-5.0** (~18% of tokens): Model is fairly wrong. Correct token is in the tail of the distribution.
- **Loss > 5.0** (~9% of tokens): Model is very surprised. The actual next token was essentially unpredicted. This is where content prediction, world knowledge, and inherently unpredictable text live.

## 5. From Loss to BPB (Bits Per Byte)

### Why Not Just Report Loss?

Cross-entropy loss is in nats per token. But different tokenizers produce different numbers of tokens for the same text. A tokenizer with 50,000 tokens produces fewer tokens (and lower per-token loss) than one with 1,024 tokens — but the model isn't actually better at compression.

**BPB (bits per byte)** normalizes for this by measuring compression quality per byte of original text, making it tokenizer-agnostic. This is the competition metric.

### The Conversion

```
bits_per_token = val_loss / ln(2)              # nats -> bits
val_bpb = bits_per_token * (total_tokens / total_bytes)
```

The critical ratio is `total_tokens / total_bytes`. For sp1024 (our tokenizer), this is approximately 1/2.45 — each token encodes about 2.45 bytes on average. So BPB is roughly `loss / ln(2) / 2.45`.

### Byte Counting (The Tricky Part)

Counting bytes per token is not trivial because of the SentencePiece **leading space convention**:

- SentencePiece represents word boundaries as `▁` (U+2581) prepended to the first token of a word
- The `▁` represents a space character (1 byte) in the original text
- But whether the `▁` counts as a byte depends on whether the **preceding token** was a boundary token

Three lookup tables are built from the tokenizer:
1. `base_bytes_lut[token_id]` — how many UTF-8 bytes this token encodes (excluding the leading space)
2. `has_leading_space_lut[token_id]` — does this token start with `▁`?
3. `is_boundary_token_lut[token_id]` — is this a control/unknown/unused token?

For each scored token:
```python
bytes = base_bytes_lut[target_token]
if has_leading_space_lut[target_token] and not is_boundary_token_lut[prev_token]:
    bytes += 1  # the ▁ represents an actual space byte
```

The space byte is attributed to the token that carries the `▁`, but only if the previous token was a "normal" token (not a control token, which would indicate a sequence boundary where no space exists).

### Concrete Example

Consider the text: `"The cat sat"`

Tokenized: `['▁The', '▁cat', '▁sat']` (token IDs: say 267, 830, 901)

- Token 267 (`▁The`): base_bytes = 3 ("The" = 3 bytes). Leading space? Yes. Previous token is boundary? Depends on context. If preceded by normal token: +1 byte = 4 total.
- Token 830 (`▁cat`): base_bytes = 3 ("cat" = 3 bytes). Leading space? Yes. Previous = 267 (normal): +1 = 4 total.
- Token 901 (`▁sat`): base_bytes = 3 ("sat" = 3 bytes). Leading space? Yes. Previous = 830 (normal): +1 = 4 total.

Total bytes: 12 (matching "The cat sat" + leading space = 11 bytes... the +1 discrepancy comes from the first token's space handling depending on broader context).

### Final BPB Computation

After processing all evaluation tokens:
```
val_loss = total_loss_sum / total_tokens     # mean CE in nats
bits_per_token = val_loss / ln(2)             # convert to bits
val_bpb = bits_per_token * (total_tokens / total_bytes)  # normalize by bytes
```

For our 13L/322 model: val_loss = 2.232, bits_per_token = 3.22, bytes/token ratio ~ 2.45, val_bpb = 3.22 / 2.45 = **1.314 BPB**.

## 6. What Context the Model Actually Has

For each prediction, the model sees:

| Position | Context tokens | Context source |
|----------|---------------|----------------|
| 0 | 0 | None — pure unigram prediction + bigram hash from position -1 (zero) |
| 1 | 1 | Previous token only + bigram |
| 10 | 10 | Short context, attention has little to work with |
| 100 | 100 | Moderate context, ~250 bytes of text |
| 512 | 512 | Good context, ~1.2KB of text |
| 1023 | 1023 | Full context, ~2.5KB of text |

**But:** the context may span unrelated documents (no document boundary markers). So position 500 might have 300 tokens from a news article followed by 200 tokens from a recipe. The model must handle this gracefully.

The **measured impact** of context: loss drops from 4.34 at position 0 to ~2.2 by position 300, then plateaus. The model extracts most value from the first ~200-300 tokens of context.

## 7. Sliding Window Evaluation (Optional)

When `val_sliding_stride > 0`, evaluation uses overlapping windows. Each window is 1024 tokens, but only the last `stride` tokens are scored. The preceding `1024 - stride` tokens provide "extra" context that the non-sliding evaluation doesn't give.

For stride=64: each scored token gets 960 tokens of left context (instead of position-dependent 0-1023). This reduces the "no context" penalty at early positions and typically improves BPB by 0.01-0.03. But it's 16x slower (1024/64 windows per token instead of 1).

The standard evaluation (non-sliding) used in this project scores every token in its natural position within the 1024-token window. This is the mode described in this document and used for all results.
