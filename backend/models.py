"""Validated public requests; coordinates always refer to original image pixels."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Placement(StrictModel):
    stamp_id: str = Field(min_length=1, max_length=100)
    x: float = Field(ge=-50000, le=100000)
    y: float = Field(ge=-50000, le=100000)
    scale: float = Field(default=1, ge=.05, le=8)
    rotation: float = Field(default=0, ge=-36000, le=36000)
    flip: bool = False


class Brush(StrictModel):
    points: list[float] = Field(min_length=4, max_length=20000)
    width: float = Field(default=12, ge=1, le=512)

    @model_validator(mode='after')
    def valid_points(self):
        if len(self.points) % 2 or any(abs(point) > 100000 for point in self.points):
            raise ValueError('Brush requires coordinate pairs within supported bounds')
        return self


class Request(StrictModel):
    image_id: str = Field(pattern='^[a-f0-9]{32}$')
    placements: list[Placement] = Field(default_factory=list, max_length=100)
    brushes: list[Brush] = Field(default_factory=list, max_length=100)
    engine: Literal['baseline', 'sdxl'] = 'baseline'
    seed: int = Field(default=0, ge=0, le=2**32-1)
    strength: float = Field(default=.65, ge=.1, le=.99)
    edit_margin: int = Field(default=24, ge=0, le=128)

    @model_validator(mode='after')
    def nonempty(self):
        if not self.placements and not self.brushes:
            raise ValueError('Place or draw a scratch before generating')
        if sum(len(brush.points) for brush in self.brushes) > 100000:
            raise ValueError('Too many brush points')
        return self


class Annotations(StrictModel):
    polygons: list[list[float]] = Field(max_length=500)

    @model_validator(mode='after')
    def bounded(self):
        if sum(len(polygon) for polygon in self.polygons) > 200000:
            raise ValueError('Too many polygon coordinates')
        return self


class Approval(StrictModel):
    confirmed: Literal[True]
