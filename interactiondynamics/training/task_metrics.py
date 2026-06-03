from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional


MetricGoal = Literal["min", "max"]


@dataclass(frozen=True)
class TaskMetricSpec:
    path: str
    goal: MetricGoal
    label: Optional[str] = None


def parse_task_metric_spec(raw: Mapping[str, Any] | None) -> Optional[TaskMetricSpec]:
    if raw is None:
        return None
    path = raw.get("path")
    goal = raw.get("goal")
    if not isinstance(path, str) or goal not in {"min", "max"}:
        return None
    label = raw.get("label")
    label_str = label if isinstance(label, str) else None
    return TaskMetricSpec(path=path, goal=goal, label=label_str)


def snapshot_metric_value(snapshot: Mapping[str, Any], path: str) -> float:
    current: Any = snapshot
    for piece in path.split("."):
        if not isinstance(current, Mapping) or piece not in current:
            return float("nan")
        current = current[piece]
    try:
        return float(current)
    except (TypeError, ValueError):
        return float("nan")


def metric_key_from_path(path: str) -> str:
    return path.rsplit(".", 1)[-1]


def is_better_metric(candidate: float, incumbent: float, goal: MetricGoal) -> bool:
    if candidate != candidate:
        return False
    if incumbent != incumbent:
        return True
    if goal == "min":
        return candidate < incumbent
    return candidate > incumbent
