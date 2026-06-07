"""Integrated Gradients attribution for a single spectrum, ONNX-only.

Reproduces scripts/interpretability/intgrad_sample.py from the 4CBLW010 repo,
but without PyTorch: the gradients ONNX Runtime can't provide are estimated by
one-sided finite differences against model_rawhead.onnx — a second export that
stops at the raw head (fc_block(_features(x)), before relu_normalize). The raw
head is what every saliency method explains, so the fibres stay independent:
on the normalised output a 2-fibre blend's curves would just mirror each other.

The maths matches intgrad_batch():
    integrate d(raw_head_f)/d(input) along the straight line from a baseline to
    the real spectrum, multiply by (x - baseline), take abs, average over the
    two input channels. Each fibre's curve is then normalised to its own max,
    exactly like save_lines_plot, so the shapes are comparable.

Cost is the price of staying torch-free: STEPS path points, each needing one
base eval plus one eval per input dimension (2*203 = 406). ~STEPS*407 forward
passes through the small raw-head graph, batched and chunked. Opt-in only —
the /predict path never touches this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.textile.inference import FIBRE_ORDER

_ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
_RAWHEAD_PATH = _ARTIFACTS / "model_rawhead.onnx"
_BASELINE_PATH = _ARTIFACTS / "baseline_mean.npy"
_WL_CUT_PATH = _ARTIFACTS / "wavelengths_cut.npy"

# Mean real spectrum in the (1, 2, 203) SNV+SG2D space — the IG script's default
# 'mean' baseline ("what makes this garment differ from an average one").
_BASELINE_MEAN = np.load(_BASELINE_PATH).astype(np.float32)        # (1, 2, 203)
# Cropped wavelength axis (nm) for the chart's x-axis.
WAVELENGTHS_NM = np.load(_WL_CUT_PATH).astype(np.float64).ravel()  # (203,)

_session = ort.InferenceSession(str(_RAWHEAD_PATH), providers=["CPUExecutionProvider"])
_INPUT_NAME = _session.get_inputs()[0].name
_OUTPUT_NAME = _session.get_outputs()[0].name

# Defaults. STEPS trades accuracy for latency; 16 is plenty for a normalised
# shape. EPS is a finite-difference step on standardised (~unit-scale) inputs.
DEFAULT_STEPS = 16
EPS = 1e-2
# Cap rows per onnxruntime call so peak memory stays a few MB on the Pi.
_CHUNK_ROWS = 2048


def _run_rawhead(batch: np.ndarray) -> np.ndarray:
    """Forward a (N, 2, 203) batch through the raw head -> (N, n_fibres)."""
    outs = []
    for i in range(0, batch.shape[0], _CHUNK_ROWS):
        chunk = np.ascontiguousarray(batch[i:i + _CHUNK_ROWS], dtype=np.float32)
        outs.append(_session.run([_OUTPUT_NAME], {_INPUT_NAME: chunk})[0])
    return np.concatenate(outs, axis=0)


def integrated_gradients(
    x: np.ndarray,
    baseline: str = "mean",
    steps: int = DEFAULT_STEPS,
) -> dict:
    """Per-fibre IG attribution for one (1, 2, 203) input.

    Returns a dict with the cropped wavelength axis (nm), the SNV spectrum (for
    a faint background trace), and a per-fibre importance curve normalised to
    [0, 1]. fibre_order matches inference.FIBRE_ORDER.
    """
    x = np.asarray(x, dtype=np.float32).reshape(1, 2, -1)
    n_bands = x.shape[-1]
    n_fibres = len(FIBRE_ORDER)

    if baseline == "zeros":
        base = np.zeros_like(x)
    elif baseline == "mean":
        base = _BASELINE_MEAN
    else:
        raise ValueError(f"unknown baseline '{baseline}', use 'mean' or 'zeros'")

    x0 = x[0]                              # (2, bands)
    base0 = base[0]                        # (2, bands)
    delta = x0 - base0                     # (2, bands)
    n_dims = x0.size                       # 2 * bands = 406

    # Right Riemann path points alpha_k = k/steps, k=1..steps (matches the script).
    alphas = np.arange(1, steps + 1, dtype=np.float32) / steps
    path = base0[None] + alphas[:, None, None] * delta[None]   # (steps, 2, bands)

    # One-sided finite differences: at each path point perturb every input dim
    # by +EPS. eye_pert[j] is EPS placed at flat index j, shaped (2, bands).
    eye_pert = (EPS * np.eye(n_dims, dtype=np.float32)).reshape(n_dims, 2, n_bands)
    perturbed = path[:, None, :, :] + eye_pert[None, :, :, :]  # (steps, n_dims, 2, bands)

    base_out = _run_rawhead(path)                                       # (steps, nf)
    pert_out = _run_rawhead(perturbed.reshape(steps * n_dims, 2, n_bands))
    pert_out = pert_out.reshape(steps, n_dims, n_fibres)               # (steps, dims, nf)

    grad = (pert_out - base_out[:, None, :]) / EPS                     # (steps, dims, nf)
    avg_grad = grad.mean(axis=0).reshape(2, n_bands, n_fibres)         # (2, bands, nf)

    # attr_f(band) = mean_channel | avg_grad * (x - baseline) |  — same as the
    # script's (g * delta).abs().mean(dim=channel).
    attr = np.abs(avg_grad * delta[:, :, None]).mean(axis=0)           # (bands, nf)

    # Normalise each fibre to its own max so curve shapes are comparable, exactly
    # as save_lines_plot does before drawing.
    importances: dict[str, list[float]] = {}
    for f, name in enumerate(FIBRE_ORDER):
        col = attr[:, f]
        m = float(col.max())
        norm = (col / m) if m > 0 else col
        importances[name] = [round(float(v), 4) for v in norm]

    spectrum = [round(float(v), 4) for v in x0[0]]      # SNV channel, faint trace
    wavelengths = [round(float(v), 2) for v in WAVELENGTHS_NM]

    return {
        "wavelengths_nm": wavelengths,
        "spectrum": spectrum,
        "fibre_order": list(FIBRE_ORDER),
        "importance": importances,
        "baseline": baseline,
        "steps": steps,
    }
