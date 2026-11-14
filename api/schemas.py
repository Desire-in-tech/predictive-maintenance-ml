"""Strict request and response schemas for predictive-maintenance inference."""
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator


class SensorReading(BaseModel):
    """One AI4I sensor observation; aliases match the original dataset columns."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    machine_type: Literal["L", "M", "H"] = Field(alias="Type")
    air_temperature: StrictInt | StrictFloat = Field(alias="Air temperature [K]")
    process_temperature: StrictInt | StrictFloat = Field(alias="Process temperature [K]")
    rotational_speed: StrictInt | StrictFloat = Field(alias="Rotational speed [rpm]")
    torque: StrictInt | StrictFloat = Field(alias="Torque [Nm]")
    tool_wear: StrictInt | StrictFloat = Field(alias="Tool wear [min]")

    @field_validator(
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    )
    @classmethod
    def require_finite_sensor_value(cls, value):
        if not math.isfinite(value):
            raise ValueError("Sensor values must be finite numbers.")
        return value


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    readings: list[SensorReading] = Field(min_length=1, max_length=5000)


class PredictionResponse(BaseModel):
    predictions: list[dict]
    count: int
