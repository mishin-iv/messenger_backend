"""Turn an uploaded spectrum CSV into the model's (1, 2, 203) input tensor.

Mirrors the training pipeline (scripts/data_preprocessing/preprocess_stacked.py
in the 4CBLW010 repo) exactly: reflectance -> absorbance -> Savitzky-Golay
(SNV channel + 2nd-derivative channel) -> crop to 1411-2536 nm -> per-channel
standardize -> stack.

Only numpy + scipy are needed here (no torch), so the Raspberry Pi stays lean.

Two CSV layouts are accepted (auto-detected):

  * long  : two columns ``wavelength,reflectance`` (one row per band), optional
            header line. Comma or semicolon separated.
  * wide  : a single spectrum spread across columns, like the project's
            ``swir_mean_spectra.csv`` -> ``Label;953.04;958.65;...`` header row
            followed by one data row. An optional leading label cell is ignored.

Reflectance is expected on the [0, 1] scale (as in training). Spectra exported
on the 0-100 percent scale are detected and rescaled.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

# ── Preprocessing parameters — must match preprocess_meta.txt / training ─────
SG_WINDOW_SNV, SG_POLY_SNV = 9, 2
SG_WINDOW_D2, SG_POLY_D2, SG_DERIV_D2 = 15, 3, 2
WL_MIN, WL_MAX = 1411.0, 2536.0          # crop window (nm)
EXPECTED_BANDS = 203                      # bands after crop -> model input width

_ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
# Canonical 288-band Specim grid the Savgol filters were tuned on. Uploads are
# resampled onto it so the filters mean the same thing regardless of the
# instrument's exact sampling.
WL_RAW = np.load(_ARTIFACTS / "wavelengths_raw.npy").astype(np.float64)


class SpectrumError(ValueError):
    """Raised when an uploaded CSV can't be parsed or doesn't cover the model's
    spectral range. The router maps this to HTTP 422."""


# ── CSV parsing ──────────────────────────────────────────────────────────────

def _maybe_float(cell: str):
    try:
        return float(cell.strip())
    except (ValueError, AttributeError):
        return None


def parse_spectrum_csv(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract (wavelengths, reflectance) from raw CSV text. Both 1-D, same len."""
    if not text or not text.strip():
        raise SpectrumError("Uploaded file is empty.")

    # Sniff the delimiter: the project uses ';', generic exports use ','.
    head = text[:4096]
    delimiter = ";" if head.count(";") >= head.count(",") else ","

    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delimiter)
            if any(c.strip() for c in r)]
    if not rows:
        raise SpectrumError("No data rows found in the CSV.")

    max_cols = max(len(r) for r in rows)

    if max_cols >= 4:
        wl, refl = _parse_wide(rows)
    else:
        wl, refl = _parse_long(rows)

    if wl.size < 10:
        raise SpectrumError(
            f"Only {wl.size} spectral points parsed; need a full spectrum."
        )
    return wl, refl


def _parse_wide(rows: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Wide layout: wavelength header row + one data row (optional label cell)."""
    header = rows[0]
    data_rows = rows[1:]
    if not data_rows:
        raise SpectrumError(
            "Wide CSV has a wavelength header but no data row beneath it."
        )
    data = data_rows[0]

    wl, refl = [], []
    for h, d in zip(header, data):
        w, r = _maybe_float(h), _maybe_float(d)
        if w is None or r is None:
            continue  # skip label/index cells (e.g. "Label" / sample name)
        wl.append(w)
        refl.append(r)
    if not wl:
        raise SpectrumError(
            "Could not read numeric wavelength/reflectance pairs from the CSV."
        )
    return np.asarray(wl, float), np.asarray(refl, float)


def _parse_long(rows: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Long layout: two columns ``wavelength,reflectance`` (header optional)."""
    wl, refl = [], []
    for r in rows:
        if len(r) < 2:
            continue
        w, v = _maybe_float(r[0]), _maybe_float(r[1])
        if w is None or v is None:
            continue  # skip header line like "wavelength,reflectance"
        wl.append(w)
        refl.append(v)
    if not wl:
        raise SpectrumError(
            "Expected two numeric columns (wavelength, reflectance) but found none."
        )
    return np.asarray(wl, float), np.asarray(refl, float)


# ── Spectrum -> model tensor ─────────────────────────────────────────────────

def _to_reflectance_fraction(refl: np.ndarray) -> np.ndarray:
    """Coerce to the [0, 1] reflectance scale used in training."""
    finite = refl[np.isfinite(refl)]
    if finite.size == 0:
        raise SpectrumError("Reflectance column contains no finite numbers.")
    hi = float(finite.max())
    if hi > 100.0:
        raise SpectrumError(
            f"Reflectance max {hi:.1f} is out of range; expected 0-1 (or 0-100%)."
        )
    if hi > 1.5:                      # clearly a 0-100 percent export
        refl = refl / 100.0
    return refl


def preprocess_spectrum(wl: np.ndarray, refl: np.ndarray) -> np.ndarray:
    """(wavelengths, reflectance) -> (1, 2, 203) float32 model input."""
    wl = np.asarray(wl, float)
    refl = _to_reflectance_fraction(np.asarray(refl, float))

    # Sort by wavelength and drop duplicate grid points (np.interp needs both).
    order = np.argsort(wl)
    wl, refl = wl[order], refl[order]
    uniq = np.concatenate(([True], np.diff(wl) > 0))
    wl, refl = wl[uniq], refl[uniq]

    if not np.all(np.isfinite(refl)):
        raise SpectrumError("Reflectance column contains non-finite values.")

    # Hard gate: the spectrum must span the model's crop window.
    if wl.min() > WL_MIN or wl.max() < WL_MAX:
        raise SpectrumError(
            f"Spectrum covers {wl.min():.0f}-{wl.max():.0f} nm but the model needs "
            f"{WL_MIN:.0f}-{WL_MAX:.0f} nm. Upload a SWIR spectrum spanning that range."
        )

    # Resample onto the canonical Specim grid the filters were tuned on.
    refl_canon = np.interp(WL_RAW, wl, refl).reshape(1, -1)

    x = _stack_channels(refl_canon, WL_RAW)
    if x.shape != (1, 2, EXPECTED_BANDS):
        raise SpectrumError(
            f"Internal preprocessing produced {x.shape}, expected (1, 2, {EXPECTED_BANDS})."
        )
    return x


def _stack_channels(x_refl: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Reflectance (1, bands) on the canonical grid -> (1, 2, n_cut) SNV+SG2D.

    Byte-for-byte identical to preprocess_row in the training/eval scripts.
    """
    X = np.log10(1.0 / np.clip(x_refl, 1e-6, 1.0))   # absorbance
    crop = (wl >= WL_MIN) & (wl <= WL_MAX)

    snv = np.asarray(savgol_filter(X, SG_WINDOW_SNV, SG_POLY_SNV, axis=1))[:, crop]
    snv = (snv - snv.mean(1, keepdims=True)) / (snv.std(1, keepdims=True) + 1e-8)

    d2 = np.asarray(savgol_filter(X, SG_WINDOW_D2, SG_POLY_D2,
                                  deriv=SG_DERIV_D2, axis=1))[:, crop]
    d2 = (d2 - d2.mean(1, keepdims=True)) / (d2.std(1, keepdims=True) + 1e-8)

    return np.stack([snv, d2], axis=1).astype(np.float32)  # (1, 2, n_cut)


def preprocess_csv_bytes(raw: bytes) -> np.ndarray:
    """Convenience: decode uploaded bytes -> (1, 2, 203) float32."""
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise SpectrumError("File is not valid text/CSV.") from exc
    wl, refl = parse_spectrum_csv(text)
    return preprocess_spectrum(wl, refl)
