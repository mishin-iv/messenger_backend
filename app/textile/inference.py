"""ONNX Runtime inference for the textile fibre-composition model.

The InferenceSession and fibre order are loaded once at import time and reused
for every request — model load is the expensive part and must not happen per
call. Only onnxruntime + numpy are required at runtime (no torch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

_ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
_MODEL_PATH = _ARTIFACTS / "model.onnx"
_FIBRE_ORDER_PATH = _ARTIFACTS / "fibre_order.txt"

# Output column order, matching the model's 10 outputs. Split on lines, not
# whitespace — "Carbon fibre" contains a space and must stay one entry.
FIBRE_ORDER: list[str] = [
    line.strip()
    for line in _FIBRE_ORDER_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

# Single shared session. CPU provider is the only one available on the Pi.
_session = ort.InferenceSession(str(_MODEL_PATH), providers=["CPUExecutionProvider"])
_INPUT_NAME = _session.get_inputs()[0].name
_OUTPUT_NAME = _session.get_outputs()[0].name


def predict(x: np.ndarray) -> dict[str, float]:
    """Run inference on a (1, 2, 203) float32 input.

    Returns a {fibre_name: fraction} dict; fractions sum to ~1.
    """
    x = np.asarray(x, dtype=np.float32)
    out = _session.run([_OUTPUT_NAME], {_INPUT_NAME: x})[0]  # (1, 10)
    comp = out[0]
    return {name: float(frac) for name, frac in zip(FIBRE_ORDER, comp)}
