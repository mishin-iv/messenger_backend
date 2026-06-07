"""FastAPI router exposing the textile fibre-composition model.

POST /api/textile/predict accepts a CSV spectrum upload and returns the
predicted 10-fibre composition. Stateless — nothing is persisted.
"""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.textile.inference import predict
from app.textile.interpretability import integrated_gradients
from app.textile.preprocessing import SpectrumError, preprocess_csv_bytes
from app.textile.schemas import SExplanation, SFibre, SPrediction

textile_router = APIRouter(prefix="/api/textile", tags=["Textile fibre composition"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # a spectrum CSV is a few tens of KB at most


def _sorted_fibres(comp: dict[str, float]) -> list[SFibre]:
    """Composition dict -> SFibre list sorted by percent, descending."""
    return sorted(
        (SFibre(fibre=name, percent=round(frac * 100, 2)) for name, frac in comp.items()),
        key=lambda f: f.percent,
        reverse=True,
    )


def _infer(raw: bytes) -> dict[str, float]:
    """CPU-bound: parse + preprocess + ONNX inference. Run in a threadpool."""
    return predict(preprocess_csv_bytes(raw))


def _explain(raw: bytes, baseline: str) -> tuple[dict[str, float], dict]:
    """CPU-bound: parse + preprocess + prediction + IG attribution."""
    x = preprocess_csv_bytes(raw)
    return predict(x), integrated_gradients(x, baseline=baseline)


@textile_router.post("/predict", response_model=SPrediction)
async def predict_composition(file: UploadFile = File(...)) -> SPrediction:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        comp = await run_in_threadpool(_infer, raw)
    except SpectrumError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    fibres = _sorted_fibres(comp)
    return SPrediction(dominant=fibres[0].fibre, composition=fibres)


@textile_router.post("/explain", response_model=SExplanation)
async def explain_composition(
    file: UploadFile = File(...),
    baseline: str = Query("mean", pattern="^(mean|zeros)$"),
) -> SExplanation:
    """Integrated-Gradients attribution for an uploaded spectrum.

    Returns the same prediction as /predict plus a per-fibre saliency curve over
    wavelength. Heavier than /predict (finite-difference gradients), so it's a
    separate opt-in call.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        comp, ig = await run_in_threadpool(_explain, raw, baseline)
    except SpectrumError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    fibres = _sorted_fibres(comp)
    return SExplanation(
        dominant=fibres[0].fibre,
        composition=fibres,
        wavelengths_nm=ig["wavelengths_nm"],
        spectrum=ig["spectrum"],
        fibre_order=ig["fibre_order"],
        importance=ig["importance"],
        baseline=ig["baseline"],
        steps=ig["steps"],
    )
