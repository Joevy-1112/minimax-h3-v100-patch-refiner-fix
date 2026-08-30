# MiniMax H3 V100 Patch — Refiner Fix

A downstream fix for [Icbears/minimax-h3-v100-patch](https://github.com/Icbears/minimax-h3-v100-patch) (v0.1.3), enabling MiniMax H3 to run with native FP16 Tensor Core compute on NVIDIA Volta cards (V100).

Upstream v0.1.3 has a coverage gap that crashes the model on ComfyUI 0.33.x. This patch closes it. **Verified end-to-end on 2×V100-16G + ComfyUI 0.33.1 + torch 2.8.0+cu126: full 124-frame 480p clip with audio, no NaN, no black frames.**

## What was broken

Icbears v0.1.3 wraps `condition_proj` to accept FP32 input (Qwen3-VL hidden states project to ~96k, overflowing FP16). But the wrapped output — still FP32 — feeds the **TokenRefiner** (the 2-block text refinement stage), whose blocks upstream never patched. The FP32 residual then hits a stock FP16-weight `qkv_proj` linear and dies:

```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half
```

All 50 main DiT blocks and the FinalLayer carry enable flags; only the refiner was missed. (The commonly reported `swiglu` crash at `model.py:183` is the same coverage gap surfacing through a different path.)

## The fix

Add an FP32-residual route for `RefinerBlock.forward`, mirroring the upstream design:

- residual stream stays FP32 (promoted on entry if needed)
- `norm1(x)` → cast to FP16 → attention → cast back to FP32 → residual add
- `norm2(x)` → cast to FP16 → MLP → cast back to FP32 → residual add
- refiner blocks receive the same enable flags as main DiT blocks (they route through the patched attention/MLP with FP32 safety islands: audio-range FP32 attention, scaled `out_proj`/`fc2`)

Also adds a lightweight numeric tracer (off by default) that logs per-tensor `max/min/nan/inf` after each stage — useful for debugging overflow points on other models/cards:

```bash
H3_FP16_TRACE=1                    # print trace lines to stdout
H3_FP16_TRACE_PATH=/tmp/trace.log  # optionally append to a file
```

No ComfyUI source file is modified. No `--fp16-unet` flag is required.

## Verified numbers (2×V100-SXM2-16G, 832×480×124 frames, euler/simple)

| Path | s/it | Note |
|---|---|---|
| INT8 convrot (eager dequant) | 55–85 | quantized weights, prior baseline |
| FP32 manual-cast | ~330 | correct but unusable |
| **FP16 + this fix** | **~29.5** | 1.9–2.9× faster than INT8; quality matches FP32 reference |

Numeric audit across 50 DiT blocks: 0 NaN / 0 Inf; largest FP16 tensor 8,744 (limit 65,504); FP32 residual stream grows 1.4e3 → 8.2e6 by design (kept off the FP16 path; branch inputs are normalized to O(10–100) before entering FP16).

## Requirements

Same as upstream v0.1.3, plus:

- ComfyUI 0.33.1+ (upstream README targets 0.33.2; structure validation auto-refuses unknown source layouts)
- torch 2.8.0+cu126 (sm_70 window; cu128+ drops Volta)
- bf16 H3 DiT weights (`minimax_h3_fl2va_pruned_bf16.safetensors`, ~38G). The patch registers FP16 as an inference dtype, so the loader casts bf16 storage to FP16 natively — no quantization, no kitchen backend.
- A dedicated ComfyUI instance for FP16 experiments (the patch must not coexist with INT8/quantized production workflows in the same process)

## Install

```bash
cd ComfyUI/custom_nodes/
git clone <this-repo> minimax-h3-v100-patch-refiner-fix
# or copy the folder; keep only one H3 patch active per instance
```

Launch ComfyUI normally. On the first H3 model load you should see:

```
[MiniMax H3 V100] v0.1.3 runtime profile installed: ...
[MiniMax H3 V100] enabled v0.1.3: ... on 50 main DiT blocks
```

If validation fails (unknown ComfyUI source structure), the patch disables itself and logs the reason — it never corrupts a model by half-applying.

## Launch flags

- `--cuda-device 0` — single-GPU mode (multi-GPU split of the 38G model is a known hang on 16G cards)
- `--disable-dynamic-vram` — avoids the `vbar_unpin` paging race documented in ComfyUI #15262
- Do **not** pass `--fp16-unet` / `--force-fp16` (upstream README rule)

## Known limits (honest version)

- Verified on V100 (sm_70) only. Ampere+ has native BF16 and should not use this; Turing would likely work but is untested.
- The 38G bf16 model exceeds 2×16G VRAM: first load pages through system RAM (~12 min with 64G RAM + 16G swap on this test box). A 32G+ card loads straight to VRAM.
- LLM-as-judge style black-frame auditing was done by frame-statistics + visual inspection, not automated perceptual metrics.
- Audio path shares the sampler; FP32 audio attention (upstream safety island) is kept but audio quality at low step counts is a separate, known H3 characteristic.

## License

GPL-3.0-only, inherited from upstream. See [NOTICE.md](NOTICE.md) for upstream attribution and modification notes.

## Credits

- [Icbears/minimax-h3-v100-patch](https://github.com/Icbears/minimax-h3-v100-patch) — the FP32-safety-island design and v0.1.3 base. This repo only closes the TokenRefiner coverage gap.
- [Amduraznak/minimax-h3-fp16-fix](https://github.com/Amduraznak/minimax-h3-fp16-fix) — the exact-math FP16 rescale approach this design builds on.
- [Comfy-Org/ComfyUI issue #15262](https://github.com/Comfy-Org/ComfyUI/issues/15262) — the residual-stream analysis that shaped the FP32-island reasoning.
