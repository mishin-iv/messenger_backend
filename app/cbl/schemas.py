from pydantic import BaseModel, Field


class SRequest(BaseModel):
    pick_up_x: float = Field(..., description="Location of pick up")
    pick_up_y: float = Field(..., description="Location of pick up")
    drop_off_x: float = Field(..., description="Location of drop off")
    drop_off_y: float = Field(..., description="Location of drop off")
    priority: int = Field(..., description="Priority of request")