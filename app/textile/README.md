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
rescaled. The spectrum **must span 1411–2536 nm** or the request is rejected
**422**. For exact fidelity, upload the full Specim SWIR range (953–2548 nm);
uploads are resampled onto that 288-band grid before filtering.

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

## Artifacts (`artifacts/`)

| File | Purpose |
|---|---|
| `model.onnx` | the model — input `(batch, 2, 203)` f32, output `(batch, 10)` summing to 1 |
| `fibre_order.txt` | the 10 output column names, in order |
| `wavelengths_raw.npy` | canonical 288-band Specim grid uploads are resampled onto |
| `preprocess_meta.txt` | Savgol/crop parameters (provenance; preprocessing.py hardcodes them) |

Regenerate from the model repo with `scripts/export_onnx.py`, then copy these
four files here.

## Runtime dependencies (Raspberry Pi)

Added to `req.txt`: `onnxruntime`, `numpy`, `scipy`. All ship prebuilt aarch64
wheels for 64-bit Raspberry Pi OS — no compiling, no PyTorch.

```bash
pip install onnxruntime numpy scipy
```
