"""Unified data schema."""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Segment:
    seg_id: str
    role: str
    kind: str
    content: str
    tokens: int = 0
    file_path: Optional[str] = None
    is_current_turn: bool = False
    turn_idx: int = -1


@dataclass
class TurnSample:
    sample_id: str
    trajectory_id: str
    turn_idx: int
    repo: str
    resolved: bool
    input_segments: list
    target_action: dict = field(default_factory=dict)
    total_input_tokens: int = 0

    def to_dict(self) -> dict:
        return asdict(self)