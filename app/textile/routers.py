"""FastAPI router exposing the textile fibre-composition model.

POST /api/textile/predict accepts a CSV spectrum upload and returns the
predicted 10-fibre composition. Stateless — nothing is persisted.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.textile.inference import predict
from app.textile.preprocessing import SpectrumError, preprocess_csv_bytes
from app.textile.schemas import SFibre, SPrediction

textile_router = APIRouter(prefix="/api/textile", tags=["Textile fibre composition"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # a spectrum CSV is a few tens of KB at most


def _infer(raw: bytes) -> dict[str, float]:
    """CPU-bound: parse + preprocess + ONNX inference. Run in a threadpool."""
    return predict(preprocess_csv_bytes(raw))


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

    fibres = sorted(
        (SFibre(fibre=name, percent=round(frac * 100, 2)) for name, frac in comp.items()),
        key=lambda f: f.percent,
        reverse=True,
    )
    return SPrediction(dominant=fibres[0].fibre, composition=fibres)
