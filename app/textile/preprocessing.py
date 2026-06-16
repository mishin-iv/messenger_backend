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
on the 0-100 percent scale are detected and rescaled. A wavenumber (cm^-1)
x-axis (common on benchtop NIR instruments) is auto-detected and converted to
nm via nm = 1e7 / cm^-1.

The model's crop window is 1411-2536 nm, but an instrument that stops short of
it (e.g. a handheld NIR capped at 1300-2350 nm) is still accepted: bands it
doesn't reach are hold-extrapolated, up to EXTRAP_MARGIN_NM per side.
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

# Range tolerance: a handheld NIR may not span the full window (e.g. one capped
# at 1300-2350 nm misses the 2350-2536 nm tail). Bands the instrument doesn't
# reach are hold-extrapolated — the boundary value is clamped outward, the same
# constant-hold edge fill the synthesis pipeline uses for NIST refs that fall
# short of the Specim grid (synthesize_blends.load_refs). Allowed only up to this
# margin per side; beyond it the model has no real signal and the upload is
# rejected. 200 nm covers the 186 nm gap of a 1300-2350 nm instrument.
EXTRAP_MARGIN_NM = 200.0

# Axis-unit detection: valid SWIR wavelengths (nm) top out near 2547 nm, while
# NIR wavenumber axes (cm^-1) covering this model's window run ~3950-12000. A
# max above this threshold cleanly marks a wavenumber axis (2547 < 3000 < 3950).
WAVENUMBER_NM_THRESHOLD = 3000.0

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


def _to_wavelength_nm(wl: np.ndarray) -> np.ndarray:
    """Convert a wavenumber (cm^-1) axis to wavelength (nm) when detected.

    NIR instruments often report cm^-1; the model expects nm. They are
    reciprocals: nm = 1e7 / cm^-1. Detection is by a max far above the SWIR
    wavelength ceiling (see WAVENUMBER_NM_THRESHOLD).
    """
    if wl.size and float(np.nanmax(wl)) > WAVENUMBER_NM_THRESHOLD:
        if np.any(wl <= 0):
            raise SpectrumError("Wavenumber axis contains non-positive values.")
        return 1e7 / wl
    return wl


def preprocess_spectrum(wl: np.ndarray, refl: np.ndarray) -> np.ndarray:
    """(wavelengths, reflectance) -> (1, 2, 203) float32 model input."""
    wl = _to_wavelength_nm(np.asarray(wl, float))
    refl = _to_reflectance_fraction(np.asarray(refl, float))

    # Sort by wavelength and drop duplicate grid points (np.interp needs both).
    order = np.argsort(wl)
    wl, refl = wl[order], refl[order]
    uniq = np.concatenate(([True], np.diff(wl) > 0))
    wl, refl = wl[uniq], refl[uniq]

    if not np.all(np.isfinite(refl)):
        raise SpectrumError("Reflectance column contains non-finite values.")

    # Coverage gate: the model wants WL_MIN..WL_MAX, but an instrument that stops
    # short is allowed as long as each side falls short by no more than
    # EXTRAP_MARGIN_NM — the uncovered bands are hold-extrapolated below. Beyond
    # that margin there's too little real signal, so reject.
    missing_low = max(0.0, wl.min() - WL_MIN)    # crop nm missing at the bottom
    missing_high = max(0.0, WL_MAX - wl.max())   # crop nm missing at the top
    if missing_low > EXTRAP_MARGIN_NM or missing_high > EXTRAP_MARGIN_NM:
        raise SpectrumError(
            f"Spectrum covers {wl.min():.0f}-{wl.max():.0f} nm but the model needs "
            f"{WL_MIN:.0f}-{WL_MAX:.0f} nm (only {EXTRAP_MARGIN_NM:.0f} nm per side "
            f"can be extrapolated). Upload a wider SWIR spectrum."
        )

    # Resample onto the canonical Specim grid the filters were tuned on. np.interp
    # clamps to the boundary value outside [wl.min, wl.max], so bands the
    # instrument never reached are filled by constant-hold extrapolation — the
    # same edge fill the synthesis pipeline uses (synthesize_blends.load_refs).
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
