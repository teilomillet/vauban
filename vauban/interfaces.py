from typing import (
    Protocol,
    List,
    Any,
    Dict,
    Optional,
    Union,
    runtime_checkable,
    AsyncGenerator,
)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ConfigDict
import uuid
import json

# --- Data Models ---


class AttackPrompt(BaseModel):
    """
    Represents a single attack attempt.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: str
    strategy: str
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: Optional[str] = None
    persona: Optional[Any] = None  # Can be a Persona object or dict
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TargetResponse(BaseModel):
    """
    Structured response from a target, supporting tool calls.
    """

    content: Optional[str] = ""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)

    def __str__(self):
        if self.tool_calls:
            tools_str = json.dumps(self.tool_calls, indent=2)
            return f"{self.content or ''}\n[TOOL CALLS]: {tools_str}"
        return self.content or ""


class JudgeResult(BaseModel):
    """
    Result of a judge's evaluation.
    """

    score: int
    label: str
    reasoning: str
    is_deceptive: bool = False


@dataclass
class SiegeResult:
    """
    Result of a Siege Campaign.
    """

    success: bool
    max_score: float
    history: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)
    termination_reason: Optional[str] = None

    @property
    def cost(self) -> float:
        return self.stats.get("estimated_cost", 0.0)

    def to_pandas(self):
        """
        Convert history to DataFrame.
        """
        import pandas as pd

        return pd.DataFrame(self.history)


# --- Protocols ---


@runtime_checkable
class Target(Protocol):
    """
    Protocol for a target model that can be attacked.
    """

    async def invoke_async(self, prompt: str) -> Union[str, TargetResponse]:
        """Send a prompt to the target and get a response."""
        ...


@runtime_checkable
class Judge(Protocol):
    """
    Protocol for assessing if a response is a breach.
    """

    async def evaluate(
        self,
        goal: str,
        prompt: str,
        response: Union[str, Any],
        success_condition: Optional[str] = None,
    ) -> JudgeResult:
        """Evaluate the target's response against the goal."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """
    Protocol for generating embeddings.
    """

    def embed(self, text: str) -> List[float]:
        """Generate an embedding vector for the text."""
        ...


@runtime_checkable
class VectorDB(Protocol):
    """
    Protocol for a vector database/memory system.
    """

    def add(self, items: List[Dict[str, Any]]) -> None:
        """Add items to the database."""
        ...

    def search(self, vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar items."""
        ...


# --- Abstract Base Classes ---


class Strategy(ABC):
    """
    Abstract base class for attack strategies.
    """

    @abstractmethod
    def generator(self) -> AsyncGenerator[List[AttackPrompt], None]:
        """
        Yields batches of attacks.
        """
        pass

    @abstractmethod
    async def update(self, results: List[Any]) -> None:
        """
        Update strategy state with results from the last batch.
        results: List of (AttackPrompt, score, response, vector)
        """
        pass


# --- Events ---

@dataclass
class Event:
    pass

@dataclass
class CampaignStartEvent(Event):
    goal: str
    max_generations: int

@dataclass
class WaveStartEvent(Event):
    generation: int

@dataclass
class AttackResultEvent(Event):
    attack: AttackPrompt
    generation: int
    response: str
    score: float
    is_breach: bool
    judge_reasoning: str
    tool_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class WaveSummaryEvent(Event):
    generation: int
    survivors: int
    best_score: float

@dataclass
class CampaignEndEvent(Event):
    result: SiegeResult
