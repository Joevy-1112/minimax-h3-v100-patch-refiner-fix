"""Minimal smoke tests for the refiner-fix distribution.

Run:  python3 test_refiner_fix.py
Requires torch; comfy imports are stubbed when ComfyUI is not on sys.path.
These tests cover the refiner-fix delta only (upstream has its own suite).
"""
import ast
import os
import sys

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append((name, detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


SRC = open(os.path.join(os.path.dirname(__file__), "runtime_patch.py"), encoding="utf-8").read()

# 1. syntax valid
try:
    ast.parse(SRC)
    check("runtime_patch.py parses", True)
except SyntaxError as e:
    check("runtime_patch.py parses", False, str(e))
    sys.exit(1)

# 2. refiner patch present
check("RefinerBlock fp32 route present", "_patched_refiner_block_forward" in SRC)
check("RefinerBlock forward hooked in install", "mm.RefinerBlock.forward = _patched_refiner_block_forward" in SRC)
check("rollback covers RefinerBlock", "mm.RefinerBlock.forward = _ORIGINAL_REFINER_BLOCK_FORWARD" in SRC)
check("refiner enable flag defined", "_REFINER_ENABLE_FLAG" in SRC)
check("refiner flags set in model init", "_REFINER_ENABLE_FLAG, True" in SRC)

# 3. fp16 cast hygiene in the refiner route (norm output must be cast before stock-weight linears)
refiner_fn = SRC.split("def _patched_refiner_block_forward", 1)[1].split("\ndef ", 1)[0]
check("refiner casts norm output to fp16", "self.norm1(x).to(dtype=torch.float16)" in refiner_fn
      and "self.norm2(x).to(dtype=torch.float16)" in refiner_fn)
check("refiner returns fp32 residual", "add_(x)" in refiner_fn and "torch.float32" in refiner_fn)

# 4. no hardcoded private paths (privacy)
for bad in ("/home/joe", "C:\\Users", "192.168.", "10.0."):
    check(f"no hardcoded path {bad!r}", bad not in SRC)

# 5. tracing is env-gated, not always-on
check("trace gated by env var", "_os.environ.get(\"H3_FP16_TRACE\")" in SRC)

# 6. fp16 dtype registration intact (upstream core mechanism untouched)
check("fp16 dtype registration intact", "supported_models.MiniMaxH3.supported_inference_dtypes = [" in SRC)
check("power-of-two scales intact", "OUT_PROJ_SCALE = 64.0" in SRC and "MLP_FC2_SCALE = 256.0" in SRC)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
