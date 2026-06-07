from pydantic import BaseModel, Field


class SFibre(BaseModel):
    fibre: str = Field(..., description="Fibre name")
    percent: float = Field(..., description="Predicted share of the garment, 0-100")


class SPrediction(BaseModel):
    dominant: str = Field(..., description="Fibre with the highest predicted share")
    composition: list[SFibre] = Field(
        ..., description="All fibres, sorted by predicted percent (descending)"
    )


class SExplanation(BaseModel):
    """Integrated-Gradients attribution for one spectrum.

    The composition mirrors /predict; importance maps each fibre name to a
    per-band saliency curve normalised to [0, 1], aligned with wavelengths_nm.
    """

    dominant: str = Field(..., description="Fibre with the highest predicted share")
    composition: list[SFibre] = Field(
        ..., description="Predicted composition, sorted by percent (descending)"
    )
    wavelengths_nm: list[float] = Field(
        ..., description="Cropped wavelength axis in nm (x-axis for every curve)"
    )
    spectrum: list[float] = Field(
        ..., description="SNV-channel spectrum, for a faint background trace"
    )
    fibre_order: list[str] = Field(..., description="Model output column order")
    importance: dict[str, list[float]] = Field(
        ..., description="Per-fibre saliency curve in [0, 1], aligned with wavelengths_nm"
    )
    baseline: str = Field(..., description="IG baseline used ('mean' or 'zeros')")
    steps: int = Field(..., description="Number of path points used for the integral")
