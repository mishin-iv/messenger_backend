from pydantic import BaseModel, Field


class SFibre(BaseModel):
    fibre: str = Field(..., description="Fibre name")
    percent: float = Field(..., description="Predicted share of the garment, 0-100")


class SPrediction(BaseModel):
    dominant: str = Field(..., description="Fibre with the highest predicted share")
    composition: list[SFibre] = Field(
        ..., description="All fibres, sorted by predicted percent (descending)"
    )
