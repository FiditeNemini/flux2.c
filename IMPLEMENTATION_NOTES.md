# FLUX.2-klein C Implementation Notes

This file tracks verified implementation details and debugging findings.
Update this as issues are found and fixed.

## Architecture Constants (Verified)
- hidden_size: 3072
- num_heads: 24
- head_dim: 128
- mlp_hidden: 9216 (3 * hidden)
- num_double_layers: 5
- num_single_layers: 20
- text_dim: 7680
- latent_channels: 128
- rope_theta: 2000.0
- axes_dim_rope: [32, 32, 32, 32] = 128 total

## RoPE Implementation (Verified)
- 4 axes: T (0-31), H (32-63), W (64-95), L (96-127)
- Image tokens: position IDs = (T=0, H=y, W=x, L=0)
  - Axis 0,3 are identity (position=0)
  - Axis 1 rotates based on y coordinate
  - Axis 2 rotates based on x coordinate
- Text tokens: position IDs = (T=0, H=0, W=0, L=seq_idx)
  - Axes 0,1,2 are identity (position=0)
  - Axis 3 rotates based on sequence index
- Rotation formula (per pair):
  - out[0] = cos*x0 - sin*x1
  - out[1] = sin*x0 + cos*x1 (NOT cos*x1 + sin*x0)

## Concatenation Order (Verified)
- Official Python concatenates as [TEXT, IMAGE] for Q, K, V
- K concatenation: cat_k = [txt_k, img_k]
- V concatenation: cat_v = [txt_v, img_v]
- RoPE PE concatenation: pe = [pe_txt, pe_img]

## Timestep Embedding (Verified)
- Input timestep is scaled by 1000 (t=1.0 becomes 1000.0)
- Sinusoidal embedding with 128 frequencies (256 dims)
- Two-layer MLP: linear_1 (256 -> 3072) + SiLU + linear_2 (3072 -> 3072)

## AdaLN Modulation (Verified)
- SiLU applied to t_emb BEFORE modulation projection
- Order: shift first, then scale: out = (1 + scale) * norm(x) + shift
- Double block: 6 params each for img/txt (shift1, scale1, gate1, shift2, scale2, gate2)
- Single block: 3 params (shift, scale, gate)

## Final Layer (Verified - BUG FIXED)
- Uses `AdaLayerNormContinuous` (not RMSNorm)
- LayerNorm with elementwise_affine=False (no learned gamma/beta)
- **CRITICAL**: Projection output splits as (scale, shift) NOT (shift, scale)
  - First half of linear output = scale
  - Second half of linear output = shift
- Formula: out = (1 + scale) * LayerNorm(x) + shift
- Linear projection to latent_channels

## Input/Output Format
- Image latent input: NCHW format [channels, h, w]
- Internal transformer: NLC format [seq, channels]
- Text input: [seq, text_dim]
- Conversion: transpose NCHW -> NLC for processing, NLC -> NCHW for output

## Verified Matching Values (Python vs C)
- t_emb values: MATCH
- img_proj (after x_embedder): MATCH
- txt_proj (after context_embedder): MATCH
- RoPE cos/sin values: MATCH
- After AdaLN in double block: MATCH
- Q values after projection and QK-norm: MATCH
- Q values after RoPE: MATCH
- Attention scores (first double block, head 0): MATCH
  - Python: [11.193116, 7.655654, 8.903099, 4.316768, 4.660241]
  - C:      [11.193114, 7.655654, 8.903103, 4.316768, 4.660240]
- Attention output (before proj): MATCH
  - Python: [-2.620135, -0.939009, 6.645398, 1.141459, 5.012253]
  - C:      [-2.620134, -0.939006, 6.645381, 1.141454, 5.012254]
- Output projection: MATCH
  - Python: [1.004017, -29.425447, -2.884938, 2.798838, -4.617723]
  - C:      [1.004027, -29.425407, -2.884941, 2.798832, -4.617741]
- After attention residual (first double block): MATCH
  - Python: [0.362979, 2.189511, 0.695620, -0.364556, 0.480292]
  - C:      [0.362981, 2.189507, 0.695621, -0.364556, 0.480294]

## ALL BUGS FIXED - Transformer output MATCHES Python!

### Verified Matching Output (2024-01-17):
- Python: [0.5482102, 2.6096351, 1.5703337, 1.7536415, 2.9919708]
- C:      [0.5482088, 2.6096334, 1.5703315, 1.7536404, 2.9919674]

All components verified matching:
- ALL 5 double blocks: MATCH
- ALL 20 single blocks: MATCH
- Final layer: MATCH

## Key Finding: Text Sequence Length
- MUST use same text sequence length in both C and Python tests
- Current test uses 512 text tokens
- Earlier mismatch was due to Python using 256 while C used 512

## Tests to Run
1. [x] Compare t_emb values
2. [x] Compare input projections
3. [x] Compare RoPE frequencies
4. [x] Compare AdaLN output
5. [x] Compare Q/K after projection and norm
6. [x] Compare Q/K after RoPE
7. [ ] Compare attention scores (first few)
8. [ ] Compare attention output
9. [ ] Compare full double block output
10. [ ] Compare single block output

## Compilation Flags
**ALWAYS use these optimized flags for faster testing:**
```bash
CFLAGS="-O3 -ffast-math -march=native"
```

## Debugging Commands
```bash
# Compile flux_transformer.c with debug output:
gcc $CFLAGS -DDEBUG_TRANSFORMER -DDEBUG_DOUBLE_BLOCK -c flux_transformer.c -o flux_transformer_debug.o

# Compile and link test:
gcc $CFLAGS -o test_tf_debug test_transformer_debug.c flux.o flux_kernels.o flux_transformer_debug.o flux_vae.o flux_safetensors.o flux_tokenizer.o flux_sample.o flux_image.o -lm

# Run C test:
./test_tf_debug flux-klein-model text_embeddings_official.bin py_noise_64x64.bin

# Run Python comparison:
python3 misc/test_diffusers_output.py
```

## IMPORTANT: After log compaction, re-read this file!
This file contains crucial implementation details that should not be forgotten.

## Bugs Fixed Summary

### Bug 1: Final Layer scale/shift order (FIXED)
- **Problem**: Final layer modulation split projection as (shift, scale)
- **Fix**: Changed to (scale, shift) - first half is scale, second half is shift
- **File**: flux_transformer.c lines 1158-1160

### Bug 2: K/V concatenation order (FIXED earlier)
- **Problem**: C code concatenated as [IMAGE, TEXT]
- **Fix**: Changed to [TEXT, IMAGE] to match Python

### Bug 3: Text sequence length mismatch (FIXED earlier)
- **Problem**: Python tests used 256 tokens, C used 512
- **Fix**: Aligned both to 512 tokens

## Single Block Architecture Reference
- 20 single-stream blocks (parallel DiT style)
- Input: concatenated [txt_hidden, img_hidden] (txt first at offset 0, img at offset txt_seq)
- Fused QKV + FFN projection
- Self-attention over full sequence
- RoPE applied: text portion uses txt_rope (axis 3), image portion uses img_rope (axes 1,2)

## VAE (AutoencoderKLFlux2) Details
- Uses `AutoencoderKLFlux2` class in diffusers
- latent_channels: 32 (VAE internal)
- patch_size: 2x2
- Transformer outputs 128 channels = 32 latent * 2*2 patch
- **Unpacking for VAE decode**: [B, 128, H, W] -> [B, 32, H*2, W*2]
  ```python
  unpacked = latents.reshape(B, 32, 2, 2, H, W)
  unpacked = unpacked.permute(0, 1, 4, 2, 5, 3)  # [B, 32, H, 2, W, 2]
  unpacked = unpacked.reshape(B, 32, H*2, W*2)
  ```
- No scaling_factor in config (unlike standard VAE)

## End-to-End Test Scripts
- `misc/generate_1step_python.py` - Generate 1-step 64x64 image with Python
  - Uses same inputs: py_noise_64x64.bin, text_embeddings_official.bin
  - Output: test_cat_python_1step.png
- C test: `./flux --dir flux-klein-model --embeddings text_embeddings_official.bin --seed 42 --steps 1 --output test_cat_fixed.png --height 64 --width 64`

## Current Status (2024-01-17)
- Transformer: FULLY WORKING (matches Python)
- VAE: Need to verify C implementation matches Python
- Next: Compare test_cat_fixed.png (C) with test_cat_python_1step.png (Python)
