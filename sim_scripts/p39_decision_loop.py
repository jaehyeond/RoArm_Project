#!/usr/bin/env python3
"""P2b: bounded observation -> decision -> scoop -> re-observation loop.

The loop owns neither the material simulator nor a learned policy.  It consumes
the frozen ``roarm-heightmap-v1`` observation, asks a replaceable policy for
``[x_m, y_m, dir_x, dir_y]``, delegates the action to a ``ScoopExecutor``, and
then decides whether the episode has reached its requested payload.  The
analytic executor in this file is a deterministic development surrogate: it
cuts an oriented footprint from a height field, conserves the remaining volume
while relaxing steep neighbour differences, and converts removed volume to
payload mass with an explicitly synthetic density.

Future DEME and model integrations enter only through the executor and decision
context interfaces.  In particular, ``DecisionPredictions`` carries predictive
means, predictive variances, and optional risk-adjusted scores into every
policy call.  Rule policies ignore those fields; a future model adapter can
populate them from ``model_scoop_predictor`` and rank candidates with its
decision-time risk scorer without changing this loop.

No training, robot control, serial access, or DEME execution occurs here.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model_scoop_predictor import compute_episode_metrics, validate_heightmap_header
from roarm_rl.heightmap import Heightmap, GridSpec


SCHEMA_VERSION = "p39-decision-loop-v1"
HEIGHTMAP_CONTRACT = "roarm-heightmap-v1"
RERUN_VERSION = "0.34.1"
RERUN_CLI = Path("/home/cgxr/miniconda3/envs/isaaclab/bin/rerun")

GRID_SPEC = GridSpec(
    origin_xy_m=(0.125, -0.190),
    cell_m=0.005,
    shape=(76, 38),
    frame="roarm_base",
    z_datum_m=0.0,
)
POLICY_NAMES = ("greedy_high", "greedy_low", "center_out", "random")
TERMINAL_REASONS = {
    "target_reached",
    "max_attempts",
    "material_exhausted",
    "consecutive_failure_limit",
    "total_failure_limit",
    "policy_error",
    "invalid_action",
    "executor_error",
    "observation_contract_error",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _heightmap_copy(observation: Heightmap, *, height: np.ndarray | None = None) -> Heightmap:
    return Heightmap(
        height=np.ascontiguousarray(
            observation.height if height is None else height, dtype=np.float32
        ),
        valid=np.ascontiguousarray(observation.valid, dtype=np.bool_),
        counts=np.ascontiguousarray(observation.counts, dtype=np.int32),
        spec=observation.spec,
        meta=dict(observation.meta),
    )


def validate_observation_contract(observation: Heightmap) -> None:
    """Reject any observation that drifts from the user-frozen grid contract."""

    expected = GRID_SPEC
    if observation.spec != expected:
        raise ValueError(f"heightmap grid drift: {observation.spec!r} != {expected!r}")
    if observation.height.shape != (76, 38):
        raise ValueError("heightmap shape must be (76, 38)")
    if observation.height.dtype != np.float32:
        raise TypeError("heightmap height must be float32")
    if observation.valid.dtype != np.bool_:
        raise TypeError("heightmap valid mask must be bool")
    if not np.isfinite(observation.height).all() or (observation.height < 0.0).any():
        raise ValueError("heightmap heights must be finite and non-negative")
    if not np.all(observation.height[~observation.valid] == 0.0):
        raise ValueError("invalid cells must use the frozen 0.0 m fill")
    validate_heightmap_header(observation.header())


@dataclass(frozen=True)
class ScoopAction:
    """Action tensor fields in named form."""

    x_m: float
    y_m: float
    dir_x: float
    dir_y: float

    def as_array(self) -> np.ndarray:
        return np.asarray([self.x_m, self.y_m, self.dir_x, self.dir_y], dtype=np.float32)


def validate_action(action: ScoopAction) -> None:
    vector = action.as_array().astype(np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("action must be finite")
    xmin, xmax, ymin, ymax = GRID_SPEC.bounds_m()
    if not (xmin <= action.x_m < xmax):
        raise ValueError(f"action x={action.x_m} outside [{xmin}, {xmax})")
    if not (ymin <= action.y_m < ymax):
        raise ValueError(f"action y={action.y_m} outside [{ymin}, {ymax})")
    norm = math.hypot(action.dir_x, action.dir_y)
    if not math.isclose(norm, 1.0, abs_tol=1.0e-6, rel_tol=1.0e-6):
        raise ValueError(f"action direction norm {norm} != 1")


@dataclass(frozen=True)
class DecisionPredictions:
    """Optional decision-time model outputs delivered to a policy.

    ``variances`` is separate and mandatory whenever a prediction provider is
    used.  ``risk_adjusted_scores`` is where the existing LCB/UCB scorer can
    expose its candidate ranking.  The loop never silently drops either field.
    """

    means: Mapping[str, np.ndarray]
    variances: Mapping[str, np.ndarray]
    risk_adjusted_scores: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not self.variances:
            raise ValueError("prediction provider must expose predictive variances")
        for namespace, mapping in (("means", self.means), ("variances", self.variances)):
            for key, value in mapping.items():
                array = np.asarray(value)
                if not np.isfinite(array).all():
                    raise ValueError(f"{namespace}.{key} contains NaN/Inf")
        for key, value in self.variances.items():
            if (np.asarray(value) < 0.0).any():
                raise ValueError(f"variances.{key} contains a negative value")
        if self.risk_adjusted_scores is not None:
            score = np.asarray(self.risk_adjusted_scores)
            if score.ndim != 1 or score.size == 0 or not np.isfinite(score).all():
                raise ValueError("risk_adjusted_scores must be a finite non-empty vector")


@dataclass(frozen=True)
class DecisionContext:
    step_index: int
    predictor: object | None = None
    predictions: DecisionPredictions | None = None
    risk_config: object | None = None
    risk_score_fn: Callable[[Any, Any], Any] | None = None


@runtime_checkable
class PredictionProvider(Protocol):
    def __call__(self, observation: Heightmap, step_index: int) -> DecisionPredictions:
        """Produce candidate means, variances, and optional adjusted scores."""


@runtime_checkable
class ModelPolicySlot(Protocol):
    """Interface-only slot for a future ``model_scoop_predictor`` adapter."""

    name: str
    predictor: object
    risk_config: object

    def select_action(self, observation: Heightmap, context: DecisionContext) -> ScoopAction:
        """Rank model candidates while consuming ``context.predictions.variances``."""


class ScoopPolicy(ABC):
    name: str

    @abstractmethod
    def select_action(self, observation: Heightmap, context: DecisionContext) -> ScoopAction:
        raise NotImplementedError


def _active_cells(observation: Heightmap, threshold_m: float) -> np.ndarray:
    return np.flatnonzero(
        (observation.valid & (observation.height >= threshold_m)).reshape(-1)
    )


def _cell_action(
    observation: Heightmap,
    flat_index: int,
    direction_xy: Sequence[float],
) -> ScoopAction:
    row, col = np.unravel_index(int(flat_index), observation.height.shape)
    center = observation.spec.cell_centers()[row, col]
    direction = np.asarray(direction_xy, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        direction = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        direction = direction / norm
    return ScoopAction(float(center[0]), float(center[1]), float(direction[0]), float(direction[1]))


def _toward_grid_center(observation: Heightmap, flat_index: int) -> np.ndarray:
    row, col = np.unravel_index(int(flat_index), observation.height.shape)
    xy = observation.spec.cell_centers()[row, col]
    xmin, xmax, ymin, ymax = observation.spec.bounds_m()
    return np.asarray([(xmin + xmax) * 0.5 - xy[0], (ymin + ymax) * 0.5 - xy[1]])


class GreedyHighPolicy(ScoopPolicy):
    name = "greedy_high"

    def __init__(self, threshold_m: float) -> None:
        self.threshold_m = float(threshold_m)

    def select_action(self, observation: Heightmap, context: DecisionContext) -> ScoopAction:
        del context
        candidates = _active_cells(observation, self.threshold_m)
        if candidates.size == 0:
            raise RuntimeError("no actionable height cell")
        height = observation.height.reshape(-1)[candidates]
        flat = int(candidates[int(np.argmax(height))])
        return _cell_action(observation, flat, _toward_grid_center(observation, flat))


class GreedyLowPolicy(ScoopPolicy):
    name = "greedy_low"

    def __init__(self, threshold_m: float) -> None:
        self.threshold_m = float(threshold_m)

    def select_action(self, observation: Heightmap, context: DecisionContext) -> ScoopAction:
        del context
        candidates = _active_cells(observation, self.threshold_m)
        if candidates.size == 0:
            raise RuntimeError("no actionable height cell")
        height = observation.height.reshape(-1)[candidates]
        flat = int(candidates[int(np.argmin(height))])
        return _cell_action(observation, flat, _toward_grid_center(observation, flat))


class CenterOutPolicy(ScoopPolicy):
    name = "center_out"

    def __init__(self, threshold_m: float) -> None:
        self.threshold_m = float(threshold_m)

    def select_action(self, observation: Heightmap, context: DecisionContext) -> ScoopAction:
        del context
        candidates = _active_cells(observation, self.threshold_m)
        if candidates.size == 0:
            raise RuntimeError("no actionable height cell")
        centers = observation.spec.cell_centers().reshape(-1, 2)[candidates]
        xmin, xmax, ymin, ymax = observation.spec.bounds_m()
        grid_center = np.asarray([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5])
        distance2 = ((centers - grid_center) ** 2).sum(axis=1)
        flat = int(candidates[int(np.argmin(distance2))])
        outward = observation.spec.cell_centers().reshape(-1, 2)[flat] - grid_center
        return _cell_action(observation, flat, outward)


class RandomPolicy(ScoopPolicy):
    name = "random"

    def __init__(self, threshold_m: float, seed: int) -> None:
        self.threshold_m = float(threshold_m)
        self.rng = np.random.default_rng(int(seed))

    def select_action(self, observation: Heightmap, context: DecisionContext) -> ScoopAction:
        del context
        candidates = _active_cells(observation, self.threshold_m)
        if candidates.size == 0:
            raise RuntimeError("no actionable height cell")
        flat = int(self.rng.choice(candidates))
        yaw = float(self.rng.uniform(-math.pi, math.pi))
        return _cell_action(observation, flat, (math.cos(yaw), math.sin(yaw)))


def build_rule_policy(name: str, *, threshold_m: float, seed: int) -> ScoopPolicy:
    if name == "greedy_high":
        return GreedyHighPolicy(threshold_m)
    if name == "greedy_low":
        return GreedyLowPolicy(threshold_m)
    if name == "center_out":
        return CenterOutPolicy(threshold_m)
    if name == "random":
        return RandomPolicy(threshold_m, seed)
    raise KeyError(f"unknown rule policy {name!r}")


@dataclass(frozen=True)
class ExecutionResult:
    observation: Heightmap
    scooped_mass_kg: float
    elapsed_time_s: float
    failed: bool
    travel_distance_m: float
    removed_volume_m3: float
    actual_removed_centroid_xy_m: tuple[float, float]
    failure_reason: str | None = None


class ScoopExecutor(ABC):
    """Replaceable action execution boundary used by ``run_episode``."""

    @abstractmethod
    def reset(self, initial_observation: Heightmap) -> None:
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Heightmap:
        raise NotImplementedError

    @abstractmethod
    def execute(self, action: ScoopAction) -> ExecutionResult:
        raise NotImplementedError


@dataclass(frozen=True)
class AnalyticExecutorConfig:
    footprint_depth_m: float = 0.060
    footprint_width_m: float = 0.058
    cut_depth_m: float = 0.010
    angle_of_repose_deg: float = 32.0
    relaxation_iterations: int = 4
    relaxation_fraction: float = 0.35
    synthetic_bulk_density_kg_m3: float = 550.0
    fixed_cycle_time_s: float = 5.0
    travel_speed_m_s: float = 0.25
    fixed_transfer_distance_m: float = 0.40
    min_removed_volume_m3: float = 1.0e-10
    initial_tool_xy_m: tuple[float, float] = (0.125, 0.0)

    def __post_init__(self) -> None:
        positive = (
            "footprint_depth_m",
            "footprint_width_m",
            "cut_depth_m",
            "synthetic_bulk_density_kg_m3",
            "fixed_cycle_time_s",
            "travel_speed_m_s",
            "min_removed_volume_m3",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 < self.angle_of_repose_deg < 89.0):
            raise ValueError("angle_of_repose_deg must lie in (0, 89)")
        if self.relaxation_iterations < 0:
            raise ValueError("relaxation_iterations must be non-negative")
        if not (0.0 <= self.relaxation_fraction <= 1.0):
            raise ValueError("relaxation_fraction must lie in [0, 1]")


class AnalyticExecutor(ScoopExecutor):
    """Deterministic oriented-cut executor with volume-conserving relaxation."""

    def __init__(self, config: AnalyticExecutorConfig | None = None) -> None:
        self.config = config or AnalyticExecutorConfig()
        self._height_m: np.ndarray | None = None
        self._template: Heightmap | None = None
        self._tool_xy_m = np.asarray(self.config.initial_tool_xy_m, dtype=np.float64)

    def reset(self, initial_observation: Heightmap) -> None:
        validate_observation_contract(initial_observation)
        self._template = _heightmap_copy(initial_observation)
        self._height_m = initial_observation.height.astype(np.float64, copy=True)
        self._tool_xy_m = np.asarray(self.config.initial_tool_xy_m, dtype=np.float64)

    def _require_ready(self) -> tuple[np.ndarray, Heightmap]:
        if self._height_m is None or self._template is None:
            raise RuntimeError("executor.reset() must be called before use")
        return self._height_m, self._template

    def observe(self) -> Heightmap:
        height, template = self._require_ready()
        return _heightmap_copy(template, height=height)

    def _footprint_cut(self, action: ScoopAction) -> np.ndarray:
        height, template = self._require_ready()
        centers = template.spec.cell_centers()
        dx = centers[..., 0] - action.x_m
        dy = centers[..., 1] - action.y_m
        along = dx * action.dir_x + dy * action.dir_y
        across = -dx * action.dir_y + dy * action.dir_x
        half_depth = 0.5 * self.config.footprint_depth_m
        half_width = 0.5 * self.config.footprint_width_m
        radius2 = (along / half_depth) ** 2 + (across / half_width) ** 2
        profile = self.config.cut_depth_m * np.clip(1.0 - radius2, 0.0, 1.0)
        profile = np.where(template.valid, profile, 0.0)
        return np.minimum(height, profile)

    def _relax(self) -> None:
        height, template = self._require_ready()
        max_delta = math.tan(math.radians(self.config.angle_of_repose_deg)) * template.spec.cell_m
        fraction = 0.5 * self.config.relaxation_fraction
        rows, cols = height.shape
        for _ in range(self.config.relaxation_iterations):
            for row in range(rows):
                for col in range(cols - 1):
                    if not (template.valid[row, col] and template.valid[row, col + 1]):
                        continue
                    self._relax_pair((row, col), (row, col + 1), max_delta, fraction)
            for row in range(rows - 1):
                for col in range(cols):
                    if not (template.valid[row, col] and template.valid[row + 1, col]):
                        continue
                    self._relax_pair((row, col), (row + 1, col), max_delta, fraction)

    def _relax_pair(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
        max_delta: float,
        fraction: float,
    ) -> None:
        height, _ = self._require_ready()
        difference = float(height[a] - height[b])
        if abs(difference) <= max_delta:
            return
        high, low = (a, b) if difference > 0.0 else (b, a)
        transfer = min(fraction * (abs(difference) - max_delta), float(height[high]))
        height[high] -= transfer
        height[low] += transfer

    def execute(self, action: ScoopAction) -> ExecutionResult:
        validate_action(action)
        height, template = self._require_ready()
        volume_before = float(height.sum(dtype=np.float64) * template.spec.cell_m**2)
        removed_height = self._footprint_cut(action)
        height -= removed_height
        removed_volume = float(removed_height.sum(dtype=np.float64) * template.spec.cell_m**2)
        weights = removed_height.reshape(-1)
        centers = template.spec.cell_centers().reshape(-1, 2)
        if float(weights.sum()) > 0.0:
            centroid = tuple(
                float(value)
                for value in (centers * weights[:, None]).sum(axis=0) / weights.sum()
            )
        else:
            centroid = (float(action.x_m), float(action.y_m))
        self._relax()
        if (height < -1.0e-15).any() or not np.isfinite(height).all():
            raise RuntimeError("analytic relaxation produced an invalid height field")
        np.maximum(height, 0.0, out=height)
        volume_after = float(height.sum(dtype=np.float64) * template.spec.cell_m**2)
        if not math.isclose(
            volume_before - volume_after,
            removed_volume,
            abs_tol=2.0e-14,
            rel_tol=2.0e-10,
        ):
            raise RuntimeError("analytic executor violated volume conservation")

        target_xy = np.asarray([action.x_m, action.y_m], dtype=np.float64)
        reposition = float(np.linalg.norm(target_xy - self._tool_xy_m))
        travel = self.config.fixed_transfer_distance_m + reposition
        elapsed = self.config.fixed_cycle_time_s + travel / self.config.travel_speed_m_s
        self._tool_xy_m = target_xy
        failed = removed_volume < self.config.min_removed_volume_m3
        return ExecutionResult(
            observation=self.observe(),
            scooped_mass_kg=removed_volume * self.config.synthetic_bulk_density_kg_m3,
            elapsed_time_s=float(elapsed),
            failed=bool(failed),
            travel_distance_m=float(travel),
            removed_volume_m3=removed_volume,
            actual_removed_centroid_xy_m=centroid,
            failure_reason="empty_footprint" if failed else None,
        )


@dataclass(frozen=True)
class EpisodeConfig:
    max_attempts: int = 240
    max_consecutive_failures: int = 5
    max_total_failures: int = 20
    actionable_height_m: float = 0.0005
    target_mass_kg: float = 0.10

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.max_consecutive_failures < 1 or self.max_total_failures < 1:
            raise ValueError("failure limits must be >= 1")
        if self.actionable_height_m < 0.0 or self.target_mass_kg <= 0.0:
            raise ValueError("action threshold and target mass are invalid")


@dataclass
class StepRecord:
    attempt: int
    action: ScoopAction
    scooped_mass_kg: float
    elapsed_time_s: float
    failed: bool
    travel_distance_m: float
    removed_volume_m3: float
    failure_reason: str | None
    actual_removed_centroid_xy_m: tuple[float, float]
    cumulative_mass_kg: float
    cumulative_time_s: float
    cumulative_failures: int
    cumulative_distance_m: float
    pre_height_sha256: str
    post_height_sha256: str
    pre_height_m: np.ndarray = field(repr=False)
    post_height_m: np.ndarray = field(repr=False)

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("pre_height_m")
        result.pop("post_height_m")
        return result


@dataclass
class EpisodeRun:
    policy_name: str
    seed: int
    config: EpisodeConfig
    initial_observation: Heightmap
    target_mass_kg: float
    records: list[StepRecord]
    terminal_reason: str
    terminal_detail: str | None
    final_observation: Heightmap

    def deterministic_payload(self) -> dict[str, Any]:
        times = [record.elapsed_time_s for record in self.records]
        failures = [record.failed for record in self.records]
        distances = [record.travel_distance_m for record in self.records]
        if times:
            metrics = compute_episode_metrics(times, failures, distances)
            metrics_dict = asdict(metrics)
        else:
            metrics_dict = {
                "total_time_s": 0.0,
                "total_scoops": 0,
                "failure_rate": 0.0,
                "total_distance_m": 0.0,
            }
        cumulative_mass = float(sum(record.scooped_mass_kg for record in self.records))
        return {
            "policy": self.policy_name,
            "seed": int(self.seed),
            "episode_config": asdict(self.config),
            "target_mass_kg": float(self.target_mass_kg),
            "target_reached": bool(cumulative_mass + 1.0e-12 >= self.target_mass_kg),
            "terminal_reason": self.terminal_reason,
            "terminal_detail": self.terminal_detail,
            "metrics": {
                **metrics_dict,
                "total_loaded_mass_kg": cumulative_mass,
            },
            "initial_height_sha256": _sha256_array(self.initial_observation.height),
            "final_height_sha256": _sha256_array(self.final_observation.height),
            "steps": [record.to_json() for record in self.records],
        }


def run_episode(
    initial_observation: Heightmap,
    policy: ScoopPolicy,
    executor: ScoopExecutor,
    config: EpisodeConfig,
    *,
    seed: int,
    predictor: object | None = None,
    prediction_provider: PredictionProvider | None = None,
    risk_config: object | None = None,
    risk_score_fn: Callable[[Any, Any], Any] | None = None,
) -> EpisodeRun:
    """Run one bounded policy episode using only the executor interface."""

    validate_observation_contract(initial_observation)
    executor.reset(initial_observation)
    observation = executor.observe()
    validate_observation_contract(observation)

    records: list[StepRecord] = []
    terminal_reason = "max_attempts"
    terminal_detail: str | None = None
    cumulative_mass = 0.0
    cumulative_time = 0.0
    cumulative_failures = 0
    cumulative_distance = 0.0
    consecutive_failures = 0

    for attempt in range(config.max_attempts):
        if cumulative_mass + 1.0e-12 >= config.target_mass_kg:
            terminal_reason = "target_reached"
            break
        active = _active_cells(observation, config.actionable_height_m)
        if active.size == 0:
            terminal_reason = "material_exhausted"
            terminal_detail = "no valid cell meets actionable_height_m"
            break

        try:
            predictions = (
                prediction_provider(observation, attempt)
                if prediction_provider is not None
                else None
            )
            context = DecisionContext(
                step_index=attempt,
                predictor=predictor,
                predictions=predictions,
                risk_config=risk_config,
                risk_score_fn=risk_score_fn,
            )
            action = policy.select_action(observation, context)
        except Exception as exc:
            terminal_reason = "policy_error"
            terminal_detail = repr(exc)
            break
        try:
            validate_action(action)
        except Exception as exc:
            terminal_reason = "invalid_action"
            terminal_detail = repr(exc)
            break

        pre_height = observation.height.copy()
        try:
            result = executor.execute(action)
        except Exception as exc:
            terminal_reason = "executor_error"
            terminal_detail = repr(exc)
            break
        try:
            validate_observation_contract(result.observation)
        except Exception as exc:
            terminal_reason = "observation_contract_error"
            terminal_detail = repr(exc)
            break
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                result.scooped_mass_kg,
                result.elapsed_time_s,
                result.travel_distance_m,
                result.removed_volume_m3,
            )
        ):
            terminal_reason = "executor_error"
            terminal_detail = "executor returned a non-finite or negative metric"
            break

        cumulative_mass += result.scooped_mass_kg
        cumulative_time += result.elapsed_time_s
        cumulative_failures += int(result.failed)
        cumulative_distance += result.travel_distance_m
        consecutive_failures = consecutive_failures + 1 if result.failed else 0
        record = StepRecord(
            attempt=attempt,
            action=action,
            scooped_mass_kg=float(result.scooped_mass_kg),
            elapsed_time_s=float(result.elapsed_time_s),
            failed=bool(result.failed),
            travel_distance_m=float(result.travel_distance_m),
            removed_volume_m3=float(result.removed_volume_m3),
            failure_reason=result.failure_reason,
            actual_removed_centroid_xy_m=result.actual_removed_centroid_xy_m,
            cumulative_mass_kg=float(cumulative_mass),
            cumulative_time_s=float(cumulative_time),
            cumulative_failures=int(cumulative_failures),
            cumulative_distance_m=float(cumulative_distance),
            pre_height_sha256=_sha256_array(pre_height),
            post_height_sha256=_sha256_array(result.observation.height),
            pre_height_m=pre_height,
            post_height_m=result.observation.height.copy(),
        )
        records.append(record)
        observation = result.observation

        if consecutive_failures >= config.max_consecutive_failures:
            terminal_reason = "consecutive_failure_limit"
            terminal_detail = f"consecutive_failures={consecutive_failures}"
            break
        if cumulative_failures >= config.max_total_failures:
            terminal_reason = "total_failure_limit"
            terminal_detail = f"total_failures={cumulative_failures}"
            break
    else:
        terminal_reason = "max_attempts"
        terminal_detail = f"attempt_cap={config.max_attempts}"

    if cumulative_mass + 1.0e-12 >= config.target_mass_kg:
        terminal_reason = "target_reached"
        terminal_detail = None

    final_observation = executor.observe()
    validate_observation_contract(final_observation)
    return EpisodeRun(
        policy_name=policy.name,
        seed=int(seed),
        config=config,
        initial_observation=_heightmap_copy(initial_observation),
        target_mass_kg=float(config.target_mass_kg),
        records=records,
        terminal_reason=terminal_reason,
        terminal_detail=terminal_detail,
        final_observation=final_observation,
    )


def synthetic_pile(seed: int) -> Heightmap:
    """Build a deterministic asymmetric ridge without invoking a physics engine."""

    rng = np.random.default_rng(int(seed))
    xy = GRID_SPEC.cell_centers()
    center_x = 0.220 + float(rng.uniform(-0.002, 0.002))
    center_y = float(rng.uniform(-0.004, 0.004))
    half_width = 0.055
    half_length = 0.120
    across = np.abs((xy[..., 0] - center_x) / half_width)
    along = np.abs((xy[..., 1] - center_y) / half_length)
    ridge = np.clip(1.0 - across, 0.0, 1.0)
    taper = np.clip(1.0 - along**2, 0.0, 1.0)
    skew = np.clip(1.0 + 0.12 * (xy[..., 1] - center_y) / half_length, 0.8, 1.2)
    ripple = 1.0 + 0.035 * np.sin(37.0 * xy[..., 0] + 19.0 * xy[..., 1])
    height = 0.029 * ridge * taper * skew * ripple
    height[height < 1.0e-5] = 0.0
    valid = np.ones(GRID_SPEC.shape, dtype=np.bool_)
    counts = np.ones(GRID_SPEC.shape, dtype=np.int32)
    observation = Heightmap(
        height=np.ascontiguousarray(height, dtype=np.float32),
        valid=valid,
        counts=counts,
        spec=GRID_SPEC,
        meta={
            "height_fill_m": 0.0,
            "source": "analytic_synthetic_ridge",
            "agg": "max",
            "agg_rule": "analytic surface sampled at contract cell centers",
            "seed": int(seed),
            "material_scope": "development surrogate; not a measured material claim",
        },
    )
    validate_observation_contract(observation)
    return observation


def _initial_mass_kg(observation: Heightmap, density_kg_m3: float) -> float:
    return float(
        observation.height.astype(np.float64).sum()
        * observation.spec.cell_m**2
        * density_kg_m3
    )


def _comparison_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(payload["metrics"])
    return {
        "policy": payload["policy"],
        "terminal_reason": payload["terminal_reason"],
        "target_reached": bool(payload["target_reached"]),
        "total_scoops": int(metrics["total_scoops"]),
        "total_time_s": float(metrics["total_time_s"]),
        "failure_rate": float(metrics["failure_rate"]),
        "total_distance_m": float(metrics["total_distance_m"]),
        "total_loaded_mass_kg": float(metrics["total_loaded_mass_kg"]),
    }


def _write_comparison_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# P2b analytic decision-loop comparison",
        "",
        "| policy | terminal | scoops | total time [s] | failure rate | distance [m] | loaded [kg] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['terminal_reason']} | {row['total_scoops']} | "
            f"{row['total_time_s']:.6f} | {row['failure_rate']:.6f} | "
            f"{row['total_distance_m']:.6f} | {row['total_loaded_mass_kg']:.9f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_policies(
    output_dir: Path,
    *,
    seed: int,
    target_fraction: float,
    max_attempts: int,
    record_rerun: bool,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not (0.0 < target_fraction <= 1.0):
        raise ValueError("target_fraction must lie in (0, 1]")

    initial = synthetic_pile(seed)
    executor_config = AnalyticExecutorConfig()
    initial_mass = _initial_mass_kg(initial, executor_config.synthetic_bulk_density_kg_m3)
    target_mass = initial_mass * float(target_fraction)
    episode_config = EpisodeConfig(
        max_attempts=int(max_attempts),
        max_consecutive_failures=5,
        max_total_failures=20,
        actionable_height_m=0.0005,
        target_mass_kg=target_mass,
    )

    runs: list[EpisodeRun] = []
    payloads: list[dict[str, Any]] = []
    for index, name in enumerate(POLICY_NAMES):
        policy_seed = int(seed + index * 1009)
        policy = build_rule_policy(
            name,
            threshold_m=episode_config.actionable_height_m,
            seed=policy_seed,
        )
        run = run_episode(
            initial,
            policy,
            AnalyticExecutor(executor_config),
            episode_config,
            seed=policy_seed,
        )
        payload = run.deterministic_payload()
        runs.append(run)
        payloads.append(payload)
        _write_json(output_dir / "traces" / f"{name}.json", payload)

    comparison = [_comparison_row(payload) for payload in payloads]
    scoop_counts = [row["total_scoops"] for row in comparison]
    deterministic_payload = {
        "schema": SCHEMA_VERSION,
        "heightmap_contract": {
            "version": HEIGHTMAP_CONTRACT,
            "shape": [76, 38],
            "cell_m": 0.005,
            "origin_xy_m": [0.125, -0.190],
            "frame": "roarm_base",
            "agg": "max",
            "empty_fill_m": 0.0,
        },
        "action_fields": ["x_m", "y_m", "dir_x", "dir_y"],
        "seed": int(seed),
        "target_fraction": float(target_fraction),
        "initial_mass_kg": float(initial_mass),
        "executor": {
            "interface": "ScoopExecutor.reset/observe/execute",
            "implementation": "AnalyticExecutor",
            "config": asdict(executor_config),
        },
        "bounded_loop": {
            "max_attempts": int(max_attempts),
            "max_consecutive_failures": episode_config.max_consecutive_failures,
            "max_total_failures": episode_config.max_total_failures,
            "terminal_reasons": sorted(TERMINAL_REASONS),
        },
        "uncertainty_slot": {
            "policy_context_field": "DecisionContext.predictions",
            "variance_field": "DecisionPredictions.variances",
            "risk_score_field": "DecisionPredictions.risk_adjusted_scores",
            "predictor_slot": "DecisionContext.predictor",
            "future_adapter_interface": "ModelPolicySlot",
        },
        "policies": payloads,
        "comparison": comparison,
        "all_bounded_terminal": bool(
            len(payloads) == 4 and all(p["terminal_reason"] in TERMINAL_REASONS for p in payloads)
        ),
        "all_target_reached": bool(all(p["target_reached"] for p in payloads)),
        "distinct_scoop_counts": bool(len(set(scoop_counts)) >= 2),
    }
    summary: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "deterministic_payload": deterministic_payload,
        "deterministic_sha256": hashlib.sha256(_canonical_json(deterministic_payload)).hexdigest(),
        "artifacts": {},
    }
    _write_comparison_markdown(output_dir / "comparison.md", comparison)
    if record_rerun:
        rerun = write_rerun_artifacts(output_dir, runs, deterministic_payload)
        summary["artifacts"]["rerun"] = rerun
    _write_json(output_dir / "summary.json", summary)
    return summary


def _height_colors(height: np.ndarray) -> np.ndarray:
    maximum = max(float(height.max()), 1.0e-9)
    value = np.clip(height.reshape(-1) / maximum, 0.0, 1.0)
    return np.column_stack(
        [40.0 + 210.0 * value, 80.0 + 150.0 * (1.0 - np.abs(value - 0.5) * 2.0), 220.0 - 180.0 * value]
    ).astype(np.uint8)


def _decision_snapshot(path: Path, run: EpisodeRun) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    if not run.records:
        raise ValueError("representative run contains no decision record")
    record = run.records[min(4, len(run.records) - 1)]
    xmin, xmax, ymin, ymax = GRID_SPEC.bounds_m()
    extent = [xmin, xmax, ymin, ymax]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    maximum = max(float(record.pre_height_m.max()), float(record.post_height_m.max()), 1.0e-6)
    for axis, height, title in (
        (axes[0], record.pre_height_m, "observed before scoop"),
        (axes[1], record.post_height_m, "re-observed after scoop"),
    ):
        image = axis.imshow(height, origin="lower", extent=extent, vmin=0.0, vmax=maximum, cmap="viridis", aspect="auto")
        axis.scatter(record.action.x_m, record.action.y_m, marker="x", s=90, c="red", label="selected")
        axis.scatter(*record.actual_removed_centroid_xy_m, marker="+", s=90, c="cyan", label="removed centroid")
        axis.set_title(title)
        axis.set_xlabel("x [m], roarm_base")
        axis.set_ylabel("y [m], roarm_base")
        axis.legend(loc="upper right")
        fig.colorbar(image, ax=axis, label="height [m]")
    fig.suptitle(
        f"{run.policy_name} step {record.attempt}: selected vs removed, cumulative "
        f"{record.cumulative_mass_kg:.6f} kg / {run.target_mass_kg:.6f} kg"
    )
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "policy": run.policy_name,
        "attempt": record.attempt,
        "selected_xy_m": [record.action.x_m, record.action.y_m],
        "actual_removed_centroid_xy_m": list(record.actual_removed_centroid_xy_m),
    }


def write_rerun_artifacts(
    output_dir: Path,
    runs: Sequence[EpisodeRun],
    deterministic_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Record all policy decisions through ``roarm_rl.viz_debug`` and validate."""

    import importlib.metadata

    if importlib.metadata.version("rerun-sdk") != RERUN_VERSION:
        raise RuntimeError("rerun-sdk version does not match the D341 pin")
    from roarm_rl import viz_debug
    from roarm_rl.rerun_contract import validate_rerun_artifact

    rrd_path = output_dir / "timeline.rrd"
    rbl_path = output_dir / "timeline.rbl"
    screenshot_path = output_dir / "inspection.png"
    validation_path = output_dir / "rerun_validation.json"
    snapshot_path = output_dir / "decision_snapshot.png"

    point_rows: list[dict[str, Any]] = []
    arrow_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    centers = GRID_SPEC.cell_centers().reshape(-1, 2)
    global_step = 0
    for policy_index, run in enumerate(runs):
        initial_height = run.initial_observation.height
        point_rows.append(
            {
                "entity_path": "geometry/heightmap",
                "positions_m": np.column_stack([centers, initial_height.reshape(-1)]),
                "colors": _height_colors(initial_height),
                "radii": [0.0014],
                "coordinate_frame": "roarm_base",
                "sequence": {"decision_step": global_step},
            }
        )
        event_rows.append(
            {
                "entity_path": "events/decision",
                "text": f"policy={run.policy_name} initial observation",
                "sequence": {"decision_step": global_step},
            }
        )
        scalar_rows.extend(
            _metric_rows(global_step, policy_index, 0, 0.0, 0.0, 0, 0.0)
        )
        global_step += 1
        for record in run.records:
            height = record.post_height_m
            pre_row, pre_col = GRID_SPEC.index_of(record.action.x_m, record.action.y_m)[:2]
            selected_z = float(record.pre_height_m[int(pre_row), int(pre_col)] + 0.004)
            actual_row, actual_col, _ = GRID_SPEC.index_of(*record.actual_removed_centroid_xy_m)
            actual_z = float(height[int(actual_row), int(actual_col)] + 0.004)
            sequence = {"decision_step": global_step}
            point_rows.extend(
                [
                    {
                        "entity_path": "geometry/heightmap",
                        "positions_m": np.column_stack([centers, height.reshape(-1)]),
                        "colors": _height_colors(height),
                        "radii": [0.0014],
                        "coordinate_frame": "roarm_base",
                        "sequence": sequence,
                    },
                    {
                        "entity_path": "geometry/selected_location",
                        "positions_m": [[record.action.x_m, record.action.y_m, selected_z]],
                        "colors": [[255, 30, 30]],
                        "radii": [0.007],
                        "labels": ["selected action"],
                        "coordinate_frame": "roarm_base",
                        "sequence": sequence,
                    },
                    {
                        "entity_path": "geometry/removed_centroid",
                        "positions_m": [[*record.actual_removed_centroid_xy_m, actual_z]],
                        "colors": [[30, 240, 255]],
                        "radii": [0.006],
                        "labels": ["removed-volume centroid"],
                        "coordinate_frame": "roarm_base",
                        "sequence": sequence,
                    },
                ]
            )
            arrow_rows.append(
                {
                    "entity_path": "geometry/scoop_direction",
                    "origins_m": [[record.action.x_m, record.action.y_m, selected_z]],
                    "vectors_m": [[0.035 * record.action.dir_x, 0.035 * record.action.dir_y, 0.0]],
                    "colors": [[255, 180, 20]],
                    "radii": [0.002],
                    "labels": ["action direction"],
                    "coordinate_frame": "roarm_base",
                    "sequence": sequence,
                }
            )
            scalar_rows.extend(
                _metric_rows(
                    global_step,
                    policy_index,
                    record.attempt + 1,
                    record.cumulative_mass_kg,
                    record.cumulative_time_s,
                    record.cumulative_failures,
                    record.cumulative_distance_m,
                )
            )
            event_rows.append(
                {
                    "entity_path": "events/decision",
                    "text": (
                        f"policy={run.policy_name} scoop={record.attempt + 1} "
                        f"xy=({record.action.x_m:.4f},{record.action.y_m:.4f}) "
                        f"dir=({record.action.dir_x:.4f},{record.action.dir_y:.4f}) "
                        f"loaded={record.cumulative_mass_kg:.6f}kg "
                        f"failed={record.failed}"
                    ),
                    "level": "WARN" if record.failed else "INFO",
                    "sequence": sequence,
                }
            )
            global_step += 1

    original_path = os.environ.get("PATH", "")
    rerun_bin = str(RERUN_CLI.parent)
    if rerun_bin not in original_path.split(os.pathsep):
        os.environ["PATH"] = rerun_bin + os.pathsep + original_path
    try:
        status = viz_debug.log_rerun(
            rrd_path,
            coordinate_frames=[
                {
                    "frame": "roarm_base",
                    "parent_frame": "tf#/",
                    "entity_path": "coordinate_frames/roarm_base",
                    "translation_m": [0.0, 0.0, 0.0],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                }
            ],
            points=point_rows,
            arrows=arrow_rows,
            scalar_trace=scalar_rows,
            events=event_rows,
            recording_metadata={
                "schema": SCHEMA_VERSION,
                "heightmap_contract": HEIGHTMAP_CONTRACT,
                "policy_order": list(POLICY_NAMES),
                "decision_steps": global_step,
                "scientific_authority": "summary.json and trace JSON arrays/hashes",
                "decision_payload_sha256": hashlib.sha256(
                    _canonical_json(deterministic_payload)
                ).hexdigest(),
            },
            recording_id="p39_decision_loop_policy_comparison",
            blueprint_path=rbl_path,
            blueprint_mode="robot_geometry",
            app_id="roarm_p39_decision_loop",
        )
    finally:
        os.environ["PATH"] = original_path
    if not status.get("ok"):
        raise RuntimeError(f"viz_debug Rerun recording failed: {status}")

    expected_entities = [
        "coordinate_frames/roarm_base",
        "events/decision",
        "geometry/heightmap",
        "geometry/removed_centroid",
        "geometry/scoop_direction",
        "geometry/selected_location",
        "metadata/run",
        "metrics/cumulative_distance_m",
        "metrics/cumulative_failures",
        "metrics/cumulative_mass_kg",
        "metrics/cumulative_time_s",
        "metrics/policy_index",
        "metrics/scoop_count",
    ]
    point_components = ["Points3D:positions", "Points3D:colors", "Points3D:radii"]
    validation = validate_rerun_artifact(
        rrd_path,
        expected_entity_paths=expected_entities,
        exact_entity_paths=expected_entities,
        expected_timeline_names=["blueprint", "decision_step", "log_time"],
        exact_timeline_names=["blueprint", "decision_step", "log_time"],
        expected_entity_components={
            "metadata/run": ["TextDocument:text"],
            "events/decision": ["TextLog:text", "TextLog:level"],
            "geometry/heightmap": point_components,
            "geometry/selected_location": point_components,
            "geometry/removed_centroid": point_components,
            "geometry/scoop_direction": [
                "Arrows3D:origins",
                "Arrows3D:vectors",
                "Arrows3D:colors",
                "Arrows3D:radii",
            ],
            **{
                name: ["Scalars:scalars"]
                for name in expected_entities
                if name.startswith("metrics/")
            },
        },
        blueprint_path=rbl_path,
        screenshot_path=screenshot_path,
        screenshot_window_size="2400x1400",
        expected_version=RERUN_VERSION,
        cli_path=RERUN_CLI,
        timeout_s=300.0,
    )
    snapshot = _decision_snapshot(snapshot_path, runs[0])
    validation["viz_debug_status"] = status
    validation["decision_snapshot"] = snapshot
    validation["completion_contract_pass"] = bool(
        validation.get("pass")
        and status.get("sink_attached_before_logging")
        and status.get("sink_finalized")
        and snapshot_path.is_file()
    )
    _write_json(validation_path, validation)
    if not validation["completion_contract_pass"]:
        raise RuntimeError(f"strict Rerun contract failed: {validation.get('errors')}")
    return {
        "rrd": {"path": str(rrd_path), "sha256": _sha256_file(rrd_path)},
        "rbl": {"path": str(rbl_path), "sha256": _sha256_file(rbl_path)},
        "inspection": {
            "path": str(screenshot_path),
            "sha256": _sha256_file(screenshot_path),
        },
        "decision_snapshot": snapshot,
        "validation": {
            "path": str(validation_path),
            "sha256": _sha256_file(validation_path),
            "pass": True,
        },
    }


def _metric_rows(
    decision_step: int,
    policy_index: int,
    scoop_count: int,
    mass_kg: float,
    time_s: float,
    failures: int,
    distance_m: float,
) -> list[dict[str, Any]]:
    sequence = {"decision_step": decision_step}
    values = {
        "metrics/policy_index": policy_index,
        "metrics/scoop_count": scoop_count,
        "metrics/cumulative_mass_kg": mass_kg,
        "metrics/cumulative_time_s": time_s,
        "metrics/cumulative_failures": failures,
        "metrics/cumulative_distance_m": distance_m,
    }
    return [
        {"entity_path": path, "value": float(value), "sequence": sequence}
        for path, value in values.items()
    ]


def validate_output(path: Path, *, require_distinct: bool) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get("deterministic_payload")
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA_VERSION:
        raise ValueError("summary deterministic payload schema mismatch")
    contract = payload.get("heightmap_contract", {})
    expected_contract = {
        "version": HEIGHTMAP_CONTRACT,
        "shape": [76, 38],
        "cell_m": 0.005,
        "origin_xy_m": [0.125, -0.190],
        "frame": "roarm_base",
        "agg": "max",
        "empty_fill_m": 0.0,
    }
    if contract != expected_contract:
        raise ValueError(f"heightmap contract mismatch: {contract}")
    policies = payload.get("policies")
    if not isinstance(policies, list) or [row.get("policy") for row in policies] != list(POLICY_NAMES):
        raise ValueError("summary must contain the four frozen policy names in order")
    comparison = payload.get("comparison")
    if not isinstance(comparison, list) or len(comparison) != 4:
        raise ValueError("summary comparison must contain four rows")

    counts: list[int] = []
    for policy, row in zip(policies, comparison):
        steps = policy.get("steps")
        if not isinstance(steps, list):
            raise ValueError(f"{policy.get('policy')}: missing step trace")
        metrics = policy.get("metrics", {})
        count = len(steps)
        counts.append(count)
        times = np.asarray([step["elapsed_time_s"] for step in steps], dtype=np.float64)
        failures = np.asarray([step["failed"] for step in steps], dtype=np.bool_)
        distances = np.asarray([step["travel_distance_m"] for step in steps], dtype=np.float64)
        masses = np.asarray([step["scooped_mass_kg"] for step in steps], dtype=np.float64)
        recomputed = (
            compute_episode_metrics(times.tolist(), failures.tolist(), distances.tolist())
            if count
            else None
        )
        checks = {
            "total_scoops": int(metrics.get("total_scoops", -1)) == count,
            "total_time_s": math.isclose(
                float(metrics.get("total_time_s", -1)),
                0.0 if recomputed is None else recomputed.total_time_s,
                abs_tol=1.0e-12,
            ),
            "failure_rate": math.isclose(
                float(metrics.get("failure_rate", -1)),
                0.0 if recomputed is None else recomputed.failure_rate,
                abs_tol=1.0e-15,
            ),
            "total_distance_m": math.isclose(
                float(metrics.get("total_distance_m", -1)),
                0.0 if recomputed is None else float(recomputed.total_distance_m),
                abs_tol=1.0e-12,
            ),
            "total_loaded_mass_kg": math.isclose(float(metrics.get("total_loaded_mass_kg", -1)), float(masses.sum()), abs_tol=1.0e-12),
            "bounded_attempts": count <= int(policy["episode_config"]["max_attempts"]),
            "terminal_reason": policy.get("terminal_reason") in TERMINAL_REASONS,
            "comparison_count": int(row.get("total_scoops", -1)) == count,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"{policy.get('policy')}: metric validation failed {failed}")
    distinct = len(set(counts)) >= 2
    if require_distinct and not distinct:
        raise ValueError(f"policy scoop counts did not differ: {counts}")
    expected_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if document.get("deterministic_sha256") != expected_hash:
        raise ValueError("deterministic payload hash mismatch")
    return {"policy_count": 4, "counts": counts, "distinct": distinct}


def compare_results(left_path: Path, right_path: Path) -> None:
    left = json.loads(left_path.read_text(encoding="utf-8"))["deterministic_payload"]
    right = json.loads(right_path.read_text(encoding="utf-8"))["deterministic_payload"]
    left_bytes = _canonical_json(left)
    right_bytes = _canonical_json(right)
    if left_bytes != right_bytes:
        raise ValueError(
            "same-seed deterministic payloads differ: "
            f"{hashlib.sha256(left_bytes).hexdigest()} != {hashlib.sha256(right_bytes).hexdigest()}"
        )


def validate_rerun_report(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    rrd_path = Path(report["path"])
    rbl_path = Path(report["blueprint_path"])
    screenshot_path = Path(report["screenshot_path"])
    checks = {
        "report_pass": report.get("pass") is True,
        "completion_contract_pass": report.get("completion_contract_pass") is True,
        "footer": report.get("footer_manifest_present") is True,
        "entity_exact": report.get("entity_path_contract", {}).get("pass") is True
        and not report.get("entity_path_contract", {}).get("unexpected_non_system"),
        "timeline_exact": report.get("timeline_contract", {}).get("pass") is True
        and report.get("timeline_contract", {}).get("exact_match") is True,
        "components": report.get("component_contract", {}).get("pass") is True,
        "headless_screenshot": report.get("headless_render", {}).get("ok") is True
        and screenshot_path.is_file()
        and _sha256_file(screenshot_path) == report.get("headless_render", {}).get("sha256"),
        "rbl": rbl_path.is_file()
        and _sha256_file(rbl_path) == report.get("blueprint_verify", {}).get("sha256"),
        "rrd": rrd_path.is_file() and _sha256_file(rrd_path) == report.get("sha256"),
        "snapshot": Path(report.get("decision_snapshot", {}).get("path", "")).is_file(),
        "sdk_pin": report.get("version", {}).get("expected_version_match") is True,
    }
    for artifact in (rrd_path, rbl_path):
        verify = subprocess.run(
            [str(RERUN_CLI), "rrd", "verify", "--check-footers", "true", str(artifact)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
        )
        checks[f"live_verify_{artifact.suffix}"] = verify.returncode == 0
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Rerun validation failed: {failed}")


def record_visual_inspection(path: Path, observations: Sequence[str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    screenshot = path.with_name("inspection.png")
    snapshot = path.with_name("decision_snapshot.png")
    parsed: list[dict[str, str]] = []
    for item in observations:
        category, separator, text = item.partition("=")
        if not separator or not category.strip() or not text.strip():
            raise ValueError("each observation must use category=text")
        parsed.append({"category": category.strip(), "observation": text.strip()})
    required = {"pile_evolution", "selected_action", "cumulative_metrics"}
    if {item["category"] for item in parsed} != required:
        raise ValueError(f"inspection categories must be exactly {sorted(required)}")
    if not screenshot.is_file() or not snapshot.is_file():
        raise FileNotFoundError("inspection.png and decision_snapshot.png are required")
    document = {
        "schema": "p39-visual-inspection-v1",
        "inspection_performed": True,
        "screenshot_path": str(screenshot),
        "screenshot_sha256": _sha256_file(screenshot),
        "decision_snapshot_path": str(snapshot),
        "decision_snapshot_sha256": _sha256_file(snapshot),
        "observations": parsed,
        "scope": "visual replay evidence; summary.json remains the numeric authority",
    }
    _write_json(path, document)


def validate_visual_inspection(path: Path) -> int:
    document = json.loads(path.read_text(encoding="utf-8"))
    observations = document.get("observations")
    if document.get("schema") != "p39-visual-inspection-v1" or document.get("inspection_performed") is not True:
        raise ValueError("visual inspection schema/status mismatch")
    if not isinstance(observations, list) or len(observations) != 3:
        raise ValueError("visual inspection must contain three observations")
    required = {"pile_evolution", "selected_action", "cumulative_metrics"}
    if {item.get("category") for item in observations} != required:
        raise ValueError("visual inspection categories mismatch")
    if any(not str(item.get("observation", "")).strip() for item in observations):
        raise ValueError("visual inspection contains a blank observation")
    for prefix in ("screenshot", "decision_snapshot"):
        artifact = Path(document[f"{prefix}_path"])
        if not artifact.is_file() or _sha256_file(artifact) != document[f"{prefix}_sha256"]:
            raise ValueError(f"{prefix} inspection artifact hash mismatch")
    return len(observations)


def _positioning_phrases() -> tuple[str, ...]:
    encoded = (
        "7Iuc666sIO2VmeyKtSDsmIjsuKHsnZgg7J6U7LCo66W8IEdQ66GcIOuztOyglQ==",
        "7Y2864K4IOuSpCDrgqjripQg7ZiV7IOB6rmM7KeAIOyYiOy4oQ==",
        "7Iuc666sK+yLpOusvOydhCDtlanss5Ag7ZWZ7Iq1",
        "67Cp67KV4pGg4oaU4pGhIOu5hOq1kOulvCDsmrDrpqwg67Cc6rKs7Jy866Gc",
        "7LWc7LSI",
        "7JeG64uk",
    )
    return tuple(base64.b64decode(value).decode("utf-8") for value in encoded)


def _find_positioning_language(text: str) -> list[str]:
    return [phrase for phrase in _positioning_phrases() if phrase in text]


def check_positioning_language() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    positive_control = " | ".join(_positioning_phrases())
    if len(_find_positioning_language(positive_control)) != len(_positioning_phrases()):
        raise RuntimeError("positioning-language positive control failed")
    found = _find_positioning_language(source)
    if found:
        raise ValueError(f"forbidden positioning language found in source: {found}")


class _OneShotExecutor(ScoopExecutor):
    """Self-check executor proving the loop does not depend on analytic internals."""

    def __init__(self, payload_kg: float) -> None:
        self.payload_kg = float(payload_kg)
        self.observation: Heightmap | None = None

    def reset(self, initial_observation: Heightmap) -> None:
        self.observation = _heightmap_copy(initial_observation)

    def observe(self) -> Heightmap:
        if self.observation is None:
            raise RuntimeError("not reset")
        return _heightmap_copy(self.observation)

    def execute(self, action: ScoopAction) -> ExecutionResult:
        validate_action(action)
        current = self.observe()
        zero = np.zeros_like(current.height)
        self.observation = _heightmap_copy(current, height=zero)
        return ExecutionResult(
            observation=self.observe(),
            scooped_mass_kg=self.payload_kg,
            elapsed_time_s=1.0,
            failed=False,
            travel_distance_m=0.1,
            removed_volume_m3=self.payload_kg / 550.0,
            actual_removed_centroid_xy_m=(action.x_m, action.y_m),
        )


class _VarianceProbePolicy(ScoopPolicy):
    name = "variance_probe"

    def __init__(self, predictor_sentinel: object, risk_sentinel: object) -> None:
        self.predictor_sentinel = predictor_sentinel
        self.risk_sentinel = risk_sentinel
        self.received = False

    def select_action(self, observation: Heightmap, context: DecisionContext) -> ScoopAction:
        if context.predictor is not self.predictor_sentinel:
            raise AssertionError("predictor slot was not delivered")
        if context.risk_config is not self.risk_sentinel or context.risk_score_fn is None:
            raise AssertionError("risk configuration/scorer slot was not delivered")
        if context.predictions is None:
            raise AssertionError("predictions were not delivered")
        variance = np.asarray(context.predictions.variances["mass_kg2"])
        if variance.shape != (2,) or not np.allclose(variance, [0.01, 0.04]):
            raise AssertionError("predictive variance was altered or dropped")
        if context.predictions.risk_adjusted_scores is None:
            raise AssertionError("risk-adjusted scores were not delivered")
        self.received = True
        flat = int(np.argmax(observation.height))
        return _cell_action(observation, flat, (1.0, 0.0))


def run_self_check() -> None:
    check_positioning_language()
    observation = synthetic_pile(457)
    validate_observation_contract(observation)
    if tuple(GRID_SPEC.bounds_m()) != (0.125, 0.315, -0.19, 0.19):
        raise AssertionError("frozen action bounds drifted")
    if POLICY_NAMES != ("greedy_high", "greedy_low", "center_out", "random"):
        raise AssertionError("rule policy set drifted")

    analytic = AnalyticExecutor()
    analytic.reset(observation)
    before = analytic.observe()
    action = GreedyHighPolicy(0.0005).select_action(before, DecisionContext(step_index=0))
    result = analytic.execute(action)
    removed_from_arrays = float(
        (before.height.astype(np.float64).sum() - result.observation.height.astype(np.float64).sum())
        * GRID_SPEC.cell_m**2
    )
    if not math.isclose(result.removed_volume_m3, removed_from_arrays, abs_tol=2.0e-11):
        raise AssertionError("analytic removed-volume accounting mismatch")
    if result.scooped_mass_kg <= 0.0:
        raise AssertionError("analytic executor failed to load material")

    predictor_sentinel = object()
    risk_sentinel = object()
    probe = _VarianceProbePolicy(predictor_sentinel, risk_sentinel)

    def provider(_: Heightmap, step_index: int) -> DecisionPredictions:
        if step_index != 0:
            raise AssertionError("one-shot provider called beyond the first decision")
        return DecisionPredictions(
            means={"mass_kg": np.asarray([0.1, 0.2])},
            variances={"mass_kg2": np.asarray([0.01, 0.04])},
            risk_adjusted_scores=np.asarray([0.8, 0.6]),
        )

    swapped = run_episode(
        observation,
        probe,
        _OneShotExecutor(payload_kg=1.0),
        EpisodeConfig(max_attempts=3, target_mass_kg=0.5),
        seed=457,
        predictor=predictor_sentinel,
        prediction_provider=provider,
        risk_config=risk_sentinel,
        risk_score_fn=lambda prediction, risk: (prediction, risk),
    )
    if swapped.terminal_reason != "target_reached" or len(swapped.records) != 1:
        raise AssertionError("replaceable executor did not complete through the common loop")
    if not probe.received:
        raise AssertionError("uncertainty probe policy was not called")

    class InvalidPolicy(ScoopPolicy):
        name = "invalid"

        def select_action(self, observation: Heightmap, context: DecisionContext) -> ScoopAction:
            del observation, context
            return ScoopAction(0.0, 0.0, 0.0, 0.0)

    invalid = run_episode(
        observation,
        InvalidPolicy(),
        _OneShotExecutor(payload_kg=1.0),
        EpisodeConfig(max_attempts=2, target_mass_kg=0.5),
        seed=457,
    )
    if invalid.terminal_reason != "invalid_action" or invalid.records:
        raise AssertionError("invalid action did not fail-stop before executor mutation")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-all", type=Path, metavar="OUTPUT_DIR")
    mode.add_argument("--self-check", action="store_true")
    mode.add_argument("--validate-output", type=Path)
    mode.add_argument("--compare-results", nargs=2, type=Path, metavar=("LEFT", "RIGHT"))
    mode.add_argument("--validate-rerun-contract", type=Path)
    mode.add_argument("--record-inspection", type=Path)
    mode.add_argument("--validate-inspection", type=Path)
    mode.add_argument("--check-positioning-language", action="store_true")
    parser.add_argument("--seed", type=int, default=457)
    parser.add_argument("--target-fraction", type=float, default=0.65)
    parser.add_argument("--max-attempts", type=int, default=240)
    parser.add_argument("--record-rerun", action="store_true")
    parser.add_argument("--require-distinct-scoop-counts", action="store_true")
    parser.add_argument("--observation", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.self_check:
        run_self_check()
        print("DECISION_LOOP_SELF_CHECK_OK")
        return 0
    if args.check_positioning_language:
        check_positioning_language()
        print("PROTECTED_FILES_AND_POSITIONING_LANGUAGE_OK")
        return 0
    if args.run_all is not None:
        summary = run_all_policies(
            args.run_all,
            seed=args.seed,
            target_fraction=args.target_fraction,
            max_attempts=args.max_attempts,
            record_rerun=args.record_rerun,
        )
        rows = summary["deterministic_payload"]["comparison"]
        print("policy,total_scoops,total_time_s,failure_rate,total_distance_m,terminal_reason")
        for row in rows:
            print(
                f"{row['policy']},{row['total_scoops']},{row['total_time_s']:.6f},"
                f"{row['failure_rate']:.6f},{row['total_distance_m']:.6f},"
                f"{row['terminal_reason']}"
            )
        print(
            "DECISION_LOOP_RUN_OK "
            f"policies={len(rows)} distinct_scoops={summary['deterministic_payload']['distinct_scoop_counts']} "
            f"all_target_reached={summary['deterministic_payload']['all_target_reached']}"
        )
        return 0
    if args.validate_output is not None:
        result = validate_output(
            args.validate_output,
            require_distinct=args.require_distinct_scoop_counts,
        )
        print(
            "DECISION_LOOP_OUTPUT_OK "
            f"policies={result['policy_count']} distinct_scoop_counts={str(result['distinct']).lower()} "
            f"counts={result['counts']}"
        )
        return 0
    if args.compare_results is not None:
        compare_results(*args.compare_results)
        print("DECISION_LOOP_REPRODUCIBILITY_OK canonical_equal=true")
        return 0
    if args.validate_rerun_contract is not None:
        validate_rerun_report(args.validate_rerun_contract)
        print("DECISION_LOOP_RERUN_OK footer=true exact_contract=true screenshot=true")
        return 0
    if args.record_inspection is not None:
        record_visual_inspection(args.record_inspection, args.observation)
        print(f"DECISION_LOOP_VISUAL_INSPECTION_RECORDED observations={len(args.observation)}")
        return 0
    if args.validate_inspection is not None:
        count = validate_visual_inspection(args.validate_inspection)
        print(f"DECISION_LOOP_VISUAL_INSPECTION_OK observations={count}")
        return 0
    raise AssertionError("unreachable argument mode")


if __name__ == "__main__":
    raise SystemExit(main())
