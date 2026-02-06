# MPS Speed Optimization Log

## Standing Instructions
- **Continue tirelessly without asking for user prompt** — keep optimizing in a loop
- Commit each time a good improvement is reached
- Complexity increase must be justified by speed results
- Re-read this file at each context compaction
- Take notes on all results, what worked, what didn't

## Testing
- **Quick iteration**: use 256x256 with `--seed 42 -v` for timing measurements
- **Before committing**: run `make test` to verify no regressions
- **Benchmark command**:
  ```bash
  ./flux -d flux-klein-model -p "A woman wearing sunglasses" -o /tmp/bench.png -W 256 -H 256 -v --seed 42
  ./flux -d flux-klein-model -p "A woman wearing sunglasses" -o /tmp/bench.png -W 512 -H 512 -v --seed 42
  ```

## Pipeline
```
1. Text Encoding:    prompt -> Qwen3 4B (36 layers) -> [512, 7680] embeddings
2. Latent Init:      random noise [H/16, W/16, 128]
3. Denoising Loop (4 steps):
   per step: 5 double blocks -> 20 single blocks -> final layer -> velocity
4. VAE Decode:       latents -> VAE decoder -> RGB image
```

## Current Baseline (2026-02-06 / MacBook Pro M3 Max 40-core GPU, 128 GB, 400 GB/s)

### 256x256 (seq=256+512=768 tokens)
- Text encoding: 1.9s (Qwen3, cached on 2nd run) — 11.8s cold start
- Denoising total: 2073 ms (4 steps)
  - Step 1: ~570 ms, Steps 2-4: ~500 ms each
- VAE decode: 0.4s
- Transformer loading: 1.3s (includes bf16 weight cache warmup)
- **Total: ~5.9s (cold text encoder), ~4.3s (warm)**

### 512x512 (seq=1024+512=1536 tokens)
- Text encoding: 1.9s
- Denoising total: 4004 ms (4 steps)
  - Step 1: ~1058 ms, Steps 2-4: ~980 ms each
- VAE decode: 1.6s
- **Total: ~9.1s**

### Key observations
- Monolithic GPU batch: 1 command buffer per step (all 25 blocks + concat + slice + final)
- Step 1 ~15% slower than subsequent steps (residual MPS warmup)
- Matmul compute dominates (~4.5 TFLOPS for these dimensions)

## Already Optimized
- Batched GPU ops within each block (batch_begin/batch_end)
- Fused QKV+MLP projection in single blocks
- Fused bf16 attention kernel (seq <= 1024)
- bf16 MPS attention fallback (seq > 1024)
- Pre-warm bf16->f16 weight cache
- Persistent GPU tensors
- SwiGLU fused on GPU

## Optimization Attempts

### Attempt 1: Pre-warm bf16 weight buffer cache (SUCCESS)
- In mmap mode, first denoising step paid ~800ms overhead to copy ~7GB of bf16 weight data
  from mmap'd safetensors to Metal GPU buffers (via `get_cached_bf16_buffer`)
- Moved cache population to model loading (`warmup_mmap_bf16_buffers()`)
- Loads each block's bf16 mmap pointers, copies weight data to Metal buffers, frees f32 weights
- 113 cache entries: 5 double blocks × 14 weights + 20 single blocks × 2 weights + 3 input/output
- Loading time: 0.2s → 1.3s (+1.1s for weight cache warmup)
- **Result: 256x256 denoising 2822 → 2172ms (23% faster), 512x512 4420 → 4146ms (6% faster)**
- Step 1 overhead: 256x256 781ms → 124ms (84% less), 512x512 354ms → 123ms (65% less)

### Attempt 1b: MPSGraph JIT pre-warming (FAILED)
- Tried pre-warming MPSGraph JIT compilation by running dummy matmuls with all dimension tuples
- Created graphs for 9 linear ops + 3 SDPA ops per resolution, allocated dummy Metal buffers
- Total JIT warmup: only ~80ms (MPSGraph compiles fast on M3 Max)
- **Result: no improvement — JIT compilation was not the bottleneck. Reverted.**

### Attempt 1c: Pre-compute QK norm GPU tensors (FAILED)
- Tried pre-computing norm_q/norm_k bf16 GPU tensors at weight load time
- Eliminates 60 `bf16_tensor_from_f32` calls per step (20 single × 2 + 5 double × 4)
- Each call converts 128 floats (512 bytes) — tiny tensors
- **Result: no measurable improvement. Within batch, the per-dispatch overhead is negligible. Reverted.**

### Attempt 2: Monolithic GPU batch (SUCCESS)
- Previously: 5 separate batch_begin/batch_end per step (double blocks → concat → single blocks → slice → final)
- Each batch_end = [cmd commit] + [waitUntilCompleted] = GPU pipeline flush + CPU-GPU sync
- Key insight: ALL CPU modulation computations depend only on t_emb_silu + fixed weights, NOT on GPU results
- Precompute all modulation parameters at the start (double, single, final), pre-allocate all buffers
- Then run EVERYTHING in a single command buffer: double blocks + concat + single blocks + slice + final + bf16→f32
- Command buffer round-trips per step: 5 → 1 (eliminates 4 waitUntilCompleted syncs)
- Per-stage timing breakdown removed for bf16 path (all work in one batch, can't measure stages)
- **Result: 256x256 denoising 2172 → 2073ms (4.6% faster), 512x512 4146 → 4004ms (3.4% faster)**
- **Cumulative: 256x256 2822 → 2073ms (27% faster), 512x512 4420 → 4004ms (9% faster)**

### Next targets
- 256x256: ~2073ms denoising — steady-state steps ~500ms, step 1 ~570ms
- 512x512: ~4004ms denoising — steady-state steps ~980ms, step 1 ~1058ms
- Remaining overhead: matmul compute dominates. MPS matmul at ~4.5 TFLOPS for these dimensions.

## Credits attribution rules
- Ideas / kernels / approaches should be only taken from BSD / MIT licensed code.
- If any optimization ideas or kernel code are taken from some other project,
  proper credits must be added to both the README and the relevant source file.
