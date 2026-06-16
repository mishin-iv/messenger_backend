# Textile fibre-composition endpoint

Serves the 4CBLW010 CNN-LSTM model (NIR/SWIR spectrum → 10-fibre composition)
as an ONNX Runtime model. No PyTorch at runtime.

## Endpoint

`POST /api/textile/predict` — `multipart/form-data`, field **`file`** = a CSV
spectrum. Stateless (nothing is stored).

**Accepted CSV layouts** (auto-detected):

- **long** — two columns `wavelength,reflectance`, one row per band, optional
  header line. Comma or semicolon separated.
- **wide** — a single spectrum across columns like `dataset/swir_mean_spectra.csv`:
  a `Label;953.04;958.65;…` header row + one data row (leading label cell ignored).

Reflectance is expected on the **0–1** scale; a 0–100 % export is detected and
rescaled. A **wavenumber (cm⁻¹) x-axis** (common on benchtop NIR instruments) is
auto-detected and converted to nm (`nm = 1e7 / cm⁻¹`). The model's window is
**1411–2536 nm**, but an instrument that stops short of it is still accepted as
long as each side falls short by no more than **200 nm** (`EXTRAP_MARGIN_NM`):
the uncovered bands are **hold-extrapolated** (the boundary value is clamped
outward, the same edge fill the synthesis pipeline uses). A handheld NIR capped
at **1300–2350 nm** works this way — the 2350–2536 nm tail is extrapolated; on
the project's `sample_19-1` this moves the prediction by only ~1.5 pp. Spectra
too short for that margin are rejected **422**. For exact fidelity, upload the
full Specim SWIR range (953–2548 nm); uploads are resampled onto that 288-band
grid before filtering.

> Note: cm⁻¹ auto-detection only fixes the axis *units* — it does not make an
> out-of-domain spectrum (different instrument/scale, e.g. the NIST NIR set)
> predict reliably.

**Response** (all 10 fibres, sorted by percent desc):

```json
{
  "dominant": "Cotton",
  "composition": [
    {"fibre": "Cotton", "percent": 52.31},
    {"fibre": "Polyester", "percent": 46.12},
    {"fibre": "Elastane", "percent": 1.40}
  ]
}
```

Errors → **422** with a `detail` message (bad/empty CSV, out-of-range, etc.);
oversized upload → **413**.

Note: Polyacrylic and Wool are absent from the training data, and Elastane /
Carbon fibre / Polyurethane sit at the sensor detection floor — treat their
values as unreliable. Cotton, Polyester and Lyocell are the strong predictions.

`POST /api/textile/explain?baseline=mean|zeros` — same upload contract, but also
returns **Integrated-Gradients** per-fibre saliency (which wavelengths drive each
fibre). Reproduces `scripts/interpretability/intgrad_sample.py` without PyTorch:
gradients are estimated by finite differences against `model_rawhead.onnx` (the
model's raw head, before `relu_normalize`, so fibres stay independent). Heavier
than `/predict` (~STEPS·407 forward passes; opt-in), validated at Pearson corr
0.9993 vs exact torch-autograd IG. `baseline` defaults to `mean` (the mean real
spectrum) — `zeros` is the SNV-zero null.

**Response** adds, alongside `dominant` + `composition`:

```json
{
  "wavelengths_nm": [1411.0, "…", 2536.0],
  "spectrum":       ["…203 SNV values for a faint background trace…"],
  "fibre_order":    ["Polyamide", "…", "Cotton"],
  "importance":     {"Cotton": ["…203 values in 0..1…"], "…": []},
  "baseline": "mean", "steps": 16
}
```

Each `importance` curve is normalised to its own peak and aligned with
`wavelengths_nm`.

## Artifacts (`artifacts/`)

| File | Purpose |
|---|---|
| `model.onnx` | the model — input `(batch, 2, 203)` f32, output `(batch, 10)` summing to 1 |
| `fibre_order.txt` | the 10 output column names, in order |
| `wavelengths_raw.npy` | canonical 288-band Specim grid uploads are resampled onto |
| `preprocess_meta.txt` | Savgol/crop parameters (provenance; preprocessing.py hardcodes them) |
| `model_rawhead.onnx` | raw head (pre-`relu_normalize`) for IG finite-diff gradients (`/explain`) |
| `baseline_mean.npy` | `(1,2,203)` mean real spectrum — the IG `mean` baseline |
| `wavelengths_cut.npy` | 203-band cropped nm axis — the `/explain` chart x-axis |

Regenerate from the model repo: `scripts/export_onnx.py` writes the first three
data files; `scripts/export_intgrad_artifacts.py` writes `model_rawhead.onnx` +
`baseline_mean.npy`. Copy all of the above here.

## Runtime dependencies (Raspberry Pi)

Added to `req.txt`: `onnxruntime`, `numpy`, `scipy`. All ship prebuilt aarch64
wheels for 64-bit Raspberry Pi OS — no compiling, no PyTorch.

```bash
pip install onnxruntime numpy scipy
```
