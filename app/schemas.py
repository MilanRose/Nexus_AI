from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    description: str
    method: Optional[str] = None
    value: Optional[Any] = None
    finding_indices: List[int] = Field(default_factory=list)
    strategy_reason: Optional[str] = None
    # Versioned profile used to retrieve relevant historical analytical
    # experiences. Kept separate from `value` so it can be queried safely.
    strategy_context: Optional[Dict[str, Any]] = None


class ModelResult(BaseModel):
    name: str
    metrics: Dict[str, float] = Field(default_factory=dict)


class AgentResult(BaseModel):
    agent: str
    dataset: str
    task: str

    findings: List[str] = Field(default_factory=list)

    # Feature names associated with this agent result
    features: List[str] = Field(default_factory=list)

    evidence: List[Evidence] = Field(default_factory=list)

    model: Optional[ModelResult] = None

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )