"""Request schemas (pydantic). Validation errors become HTTP 422 with field-level messages."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SORTABLE = ("rank_exposure", "fraud_probability", "total_reimbursed", "estimated_exposure", "total_claims",
            "unique_beneficiaries", "Provider")
Status = Literal["not_reviewed", "under_review", "escalated", "closed"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderListQuery(_Strict):
    page: int = Field(1, ge=1)
    per_page: int = Field(25, ge=1, le=100)
    sort: Literal[SORTABLE] = "rank_exposure"           # type: ignore[valid-type]
    order: Literal["asc", "desc"] = "asc"
    search: Optional[str] = Field(None, max_length=32)
    risk_level: Optional[Literal["high", "medium", "low"]] = None
    priority: Optional[Literal["P1", "P2", "P3", "Routine"]] = None
    status: Optional[Status] = None
    min_probability: Optional[float] = Field(None, ge=0, le=1)
    max_probability: Optional[float] = Field(None, ge=0, le=1)
    min_exposure: Optional[float] = Field(None, ge=0)
    max_exposure: Optional[float] = Field(None, ge=0)
    within_capacity: Optional[int] = Field(None, ge=1)
    evaluation_mode: Literal["operational", "retrospective"] = "operational"

    @field_validator("search")
    @classmethod
    def _strip(cls, v):
        return v.strip() or None if v is not None else None

    @model_validator(mode="after")
    def _ranges(self):
        if self.min_probability is not None and self.max_probability is not None and self.min_probability > self.max_probability:
            raise ValueError("min_probability cannot exceed max_probability")
        if self.min_exposure is not None and self.max_exposure is not None and self.min_exposure > self.max_exposure:
            raise ValueError("min_exposure cannot exceed max_exposure")
        return self


class CapacityQuery(_Strict):
    capacity: int = Field(100, ge=1, le=100000)
    include_curves: bool = False


class ProviderDetailQuery(_Strict):
    evaluation_mode: Literal["operational", "retrospective"] = "operational"


class StatusUpdate(_Strict):
    status: Status
    note: Optional[str] = Field(None, max_length=2000)


class PredictRequest(_Strict):
    features: dict[str, Optional[float]] = Field(..., min_length=1, max_length=200)
