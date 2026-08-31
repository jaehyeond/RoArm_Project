"""Scoop outcome predictor skeleton and its versioned data contract.

This file deliberately stops at contracts, forward-pass plumbing, uncertainty
outputs, decision-time uncertainty use, and rollout metrics.  It does not load
data, generate data, optimize parameters, or control the robot.

Literature-positioning guard
----------------------------
The two predictor modes below operationalize an established experimental
premise; this module makes no novelty claim about that comparison or about any
calibration architecture.  The design-specific requirement enforced here is
that uncertainty is not merely logged: :func:`risk_adjusted_candidate_score`
uses every available predictive variance when candidate actions are ranked.

DATA CONTRACT (``scoop-predictor-v1``)
======================================

Observation
-----------
The producer contract is ``roarm-heightmap-v1`` from
``roarm_rl.heightmap`` (the concurrent s2 heightmap work):

* ``height_m``: ``[B, 1, 76, 38]`` ``torch.float32``, metres, expected range
  ``[0, +inf)``.  Values are surface z in ``roarm_base`` minus the support-plane
  datum.  Row indexes +y, column indexes +x.  Cell size is 0.005 m and the
  lower-left corner of cell [0, 0] is (0.125, -0.190) m.  The grid spans
  0.190 m in x and 0.380 m in y, matching the s1 DEME pile box
  (186.6 x 377.5 mm, ridge ratio 2.18:1).  Chosen 2026-08-31 by the user:
  it puts mouth-width / pile-width at 58/377.5 = 0.154, near the 0.05-0.1 of
  real industrial grabs, instead of the 0.53 the earlier 150 mm window gave.
* ``valid_mask``: ``[B, 1, 76, 38]`` ``torch.bool``.  False means the height
  value is padding, not a measurement.  The mask is a separate CNN channel.
* The model-v1 header requires ``agg="max"``: each height is the highest
  surface point in the cell's square footprint, and an invalid cell is filled
  with 0.0 m.  Producer source (particles or depth) may differ, but aggregation
  must not; changing it requires a predictor-contract version change.
* The producer's ``counts`` array remains an int32 diagnostic and is excluded
  from model input: particle and depth producers give it different sampling
  semantics, so consuming it would introduce a source-domain shortcut.

Action
------
``action`` is ``[B, 4]`` ``torch.float32`` with
``[x_m, y_m, dir_x, dir_y]``:

* ``x_m`` is the scoop-mouth centre in ``roarm_base`` x, half-open range
  ``[0.125, 0.315)`` m for this grid.
* ``y_m`` is the scoop-mouth centre in ``roarm_base`` y, half-open range
  ``[-0.190, 0.190)`` m for this grid.

  .. warning::
     The placement of this box in ``roarm_base`` is NOT yet reach-verified.
     x starts at 0.125 m to match the earlier window, and y is centred on the
     base, but nobody has checked that the arm can reach x=0.315 m or
     y=+-0.190 m with the scoop pose.  Verify against JOINT_LIMITS and the
     fixed path before any real-robot or reach-dependent use.
* ``(dir_x, dir_y)`` is a unit vector in the base XY plane, each component in
  ``[-1, 1]``.  It represents yaw without an angle-wrap discontinuity;
  ``yaw = atan2(dir_y, dir_x)`` and has range ``[-pi, pi)`` radians.

The only free action variables are x, y, and yaw.  Approach height, insertion
depth, tool pitch, speed profile, closing schedule, lift, transfer, dump pose,
and release are the externally executed fixed path
``scoop_v0_fixed_path_v1``.  They are intentionally absent from the tensor.
Changing any of them requires a new trajectory contract id; samples from two
path ids must not be silently mixed.

Supervision targets
-------------------
All continuous targets are ``torch.float32``:

* ``scooped_mass_kg``: ``[B, 1]``, kg, range ``[0, +inf)``.
* ``elapsed_time_s``: ``[B, 1]``, seconds, range ``[0, +inf)``.  This is the
  measured time for the full fixed path, including a failed attempt.
* ``next_height_m``: ``[B, 1, 76, 38]``, metres, range ``[0, +inf)``; required
  only in ``amount_and_shape`` mode.  ``next_valid_mask`` has the same shape
  and dtype ``torch.bool`` and masks unavailable supervision cells.
* ``failed``: ``[B, 1]`` ``torch.bool``.  The concrete failure predicate must
  be frozen in each dataset manifest; the predictor emits its probability.

Prediction and uncertainty
--------------------------
``ScoopPrediction`` returns mean and variance together:

* mass mean ``[B, 1]`` kg and variance ``[B, 1]`` kg^2;
* time mean ``[B, 1]`` s and variance ``[B, 1]`` s^2;
* failure probability ``[B, 1]`` in ``[0, 1]`` and Bernoulli variance
  ``p(1-p)`` ``[B, 1]`` (dimensionless);
* next-height mean ``[B, 1, 76, 38]`` m and variance of the same shape in m^2
  in ``amount_and_shape`` mode.  Both fields are ``None`` in ``amount_only``
  mode, while the enclosing return type and forward signature stay identical.

The positive variance heads are model placeholders until data exist.  A later
probabilistic calibration component may replace or augment them, but the
decision interface remains the same: lower-confidence mass, upper-confidence
time/failure, and (when present) height standard deviation enter the candidate
score.  Thus a variance-producing component cannot be bypassed without an
explicit decision-layer change.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PREDICTOR_CONTRACT_VERSION = "scoop-predictor-v1"
HEIGHTMAP_CONTRACT_VERSION = "roarm-heightmap-v1"
FIXED_TRAJECTORY_CONTRACT_ID = "scoop_v0_fixed_path_v1"


class PredictionMode(str, Enum):
    """The pre-registered amount-only versus amount-and-shape switch."""

    AMOUNT_ONLY = "amount_only"
    AMOUNT_AND_SHAPE = "amount_and_shape"


@dataclass(frozen=True)
class HeightmapGridContract:
    """Frozen s2 heightmap geometry consumed by this model."""

    spec_version: str = HEIGHTMAP_CONTRACT_VERSION
    frame: str = "roarm_base"
    shape: tuple[int, int] = (76, 38)
    cell_m: float = 0.005
    origin_xy_m: tuple[float, float] = (0.125, -0.190)
    z_datum_m: float = 0.0
    aggregation: str = "max"
    empty_fill_m: float = 0.0
    indexing: str = "height[row, col]; row -> +y, col -> +x"

    @property
    def bounds_xy_m(self) -> tuple[float, float, float, float]:
        """Return inclusive action-domain bounds (xmin, xmax, ymin, ymax)."""

        rows, cols = self.shape
        x0, y0 = self.origin_xy_m
        return x0, x0 + cols * self.cell_m, y0, y0 + rows * self.cell_m


@dataclass(frozen=True)
class ScoopDataContract:
    """Versioned linkage between observations, actions, and the fixed path."""

    version: str = PREDICTOR_CONTRACT_VERSION
    heightmap: HeightmapGridContract = HeightmapGridContract()
    trajectory_contract_id: str = FIXED_TRAJECTORY_CONTRACT_ID
    action_fields: tuple[str, ...] = ("x_m", "y_m", "dir_x", "dir_y")
    amount_unit: str = "kg"
    time_unit: str = "s"


DEFAULT_DATA_CONTRACT = ScoopDataContract()


def validate_heightmap_header(
    header: Mapping[str, object],
    contract: ScoopDataContract = DEFAULT_DATA_CONTRACT,
) -> None:
    """Validate source metadata before converting numpy arrays to tensors."""

    grid = contract.heightmap
    exact_expected = {
        "spec_version": grid.spec_version,
        "frame": grid.frame,
        "agg": grid.aggregation,
        "indexing": grid.indexing,
    }
    for key, expected in exact_expected.items():
        if header.get(key) != expected:
            raise ValueError(f"heightmap header {key}={header.get(key)!r} != {expected!r}")

    if tuple(header.get("shape", ())) != grid.shape:
        raise ValueError(f"heightmap header shape must be {grid.shape}")
    origin = tuple(header.get("origin_xy_m", ()))
    if len(origin) != 2 or any(
        abs(float(actual) - expected) > 1.0e-12
        for actual, expected in zip(origin, grid.origin_xy_m)
    ):
        raise ValueError(f"heightmap header origin_xy_m must be {grid.origin_xy_m}")
    for key, expected in (("cell_m", grid.cell_m), ("z_datum_m", grid.z_datum_m)):
        value = header.get(key)
        if value is None or abs(float(value) - expected) > 1.0e-12:
            raise ValueError(f"heightmap header {key} must be {expected}")

    empty = header.get("empty_cell")
    if not isinstance(empty, Mapping):
        raise ValueError("heightmap header empty_cell must be a mapping")
    fill = empty.get("height_fill_m")
    if fill is None or abs(float(fill) - grid.empty_fill_m) > 1.0e-12:
        raise ValueError(
            f"heightmap header empty_cell.height_fill_m must be {grid.empty_fill_m}"
        )


@dataclass
class ScoopTarget:
    """Training/evaluation target container; no training loop is implemented."""

    scooped_mass_kg: Tensor
    elapsed_time_s: Tensor
    failed: Tensor
    next_height_m: Tensor | None = None
    next_valid_mask: Tensor | None = None


@dataclass
class ScoopPrediction:
    """Common output type for both predictor modes."""

    mode: PredictionMode
    scooped_mass_mean_kg: Tensor
    scooped_mass_variance_kg2: Tensor
    elapsed_time_mean_s: Tensor
    elapsed_time_variance_s2: Tensor
    failure_probability: Tensor
    failure_variance: Tensor
    next_height_mean_m: Tensor | None
    next_height_variance_m2: Tensor | None


@dataclass(frozen=True)
class PredictorConfig:
    """Architecture constants only; these are not fitted hyperparameters."""

    mode: PredictionMode | str = PredictionMode.AMOUNT_AND_SHAPE
    spatial_channels: int = 32
    action_features: int = 32
    hidden_features: int = 128
    height_scale_m: float = 0.10
    mass_scale_kg: float = 0.02
    time_scale_s: float = 5.0
    height_delta_limit_m: float = 0.10
    min_mass_variance_kg2: float = 1.0e-10
    min_time_variance_s2: float = 1.0e-6
    min_height_variance_m2: float = 1.0e-10

    def resolved_mode(self) -> PredictionMode:
        return PredictionMode(self.mode)


def _require_shape(name: str, value: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {tuple(value.shape)} != {expected}")


def _require_float32(name: str, value: Tensor) -> None:
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")


def validate_model_inputs(
    height_m: Tensor,
    valid_mask: Tensor,
    action: Tensor,
    contract: ScoopDataContract = DEFAULT_DATA_CONTRACT,
) -> None:
    """Fail fast when a batch crosses a unit, shape, dtype, or action boundary."""

    if height_m.ndim != 4:
        raise ValueError(f"height_m must be [B, 1, H, W], got {height_m.shape}")
    batch = height_m.shape[0]
    rows, cols = contract.heightmap.shape
    expected_map = (batch, 1, rows, cols)
    _require_shape("height_m", height_m, expected_map)
    _require_shape("valid_mask", valid_mask, expected_map)
    _require_shape("action", action, (batch, 4))
    _require_float32("height_m", height_m)
    _require_float32("action", action)
    if valid_mask.dtype != torch.bool:
        raise TypeError(f"valid_mask must be torch.bool, got {valid_mask.dtype}")
    if height_m.device != valid_mask.device or height_m.device != action.device:
        raise ValueError("height_m, valid_mask, and action must share one device")
    if not torch.isfinite(height_m).all():
        raise ValueError("height_m must be finite; invalid cells use fill + valid_mask")
    if not torch.isfinite(action).all():
        raise ValueError("action must be finite")
    if (height_m < 0).any():
        raise ValueError("height_m must be support-relative and non-negative")

    xmin, xmax, ymin, ymax = contract.heightmap.bounds_xy_m
    if ((action[:, 0] < xmin) | (action[:, 0] >= xmax)).any():
        raise ValueError(f"action x_m must lie in [{xmin}, {xmax}) m")
    if ((action[:, 1] < ymin) | (action[:, 1] >= ymax)).any():
        raise ValueError(f"action y_m must lie in [{ymin}, {ymax}) m")
    direction_norm = torch.linalg.vector_norm(action[:, 2:4], dim=1)
    if not torch.allclose(
        direction_norm,
        torch.ones_like(direction_norm),
        atol=1.0e-4,
        rtol=1.0e-4,
    ):
        raise ValueError("(dir_x, dir_y) must be a unit vector")


def validate_targets(
    target: ScoopTarget,
    mode: PredictionMode | str,
    contract: ScoopDataContract = DEFAULT_DATA_CONTRACT,
) -> None:
    """Validate target tensors before a future loss function consumes them."""

    resolved_mode = PredictionMode(mode)
    if target.scooped_mass_kg.ndim != 2:
        raise ValueError("scooped_mass_kg must be [B, 1]")
    batch = target.scooped_mass_kg.shape[0]
    scalar_shape = (batch, 1)
    _require_shape("scooped_mass_kg", target.scooped_mass_kg, scalar_shape)
    _require_shape("elapsed_time_s", target.elapsed_time_s, scalar_shape)
    _require_shape("failed", target.failed, scalar_shape)
    _require_float32("scooped_mass_kg", target.scooped_mass_kg)
    _require_float32("elapsed_time_s", target.elapsed_time_s)
    if target.failed.dtype != torch.bool:
        raise TypeError(f"failed must be torch.bool, got {target.failed.dtype}")
    for name, value in (
        ("scooped_mass_kg", target.scooped_mass_kg),
        ("elapsed_time_s", target.elapsed_time_s),
    ):
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"{name} must be finite and non-negative")

    if resolved_mode is PredictionMode.AMOUNT_ONLY:
        if target.next_height_m is not None or target.next_valid_mask is not None:
            raise ValueError("amount_only targets must omit next-height tensors")
        return

    if target.next_height_m is None or target.next_valid_mask is None:
        raise ValueError("amount_and_shape targets require height and validity tensors")
    rows, cols = contract.heightmap.shape
    map_shape = (batch, 1, rows, cols)
    _require_shape("next_height_m", target.next_height_m, map_shape)
    _require_shape("next_valid_mask", target.next_valid_mask, map_shape)
    _require_float32("next_height_m", target.next_height_m)
    if target.next_valid_mask.dtype != torch.bool:
        raise TypeError("next_valid_mask must be torch.bool")
    if not torch.isfinite(target.next_height_m).all() or (target.next_height_m < 0).any():
        raise ValueError("next_height_m must be finite and non-negative")


class ScoopPredictor(nn.Module):
    """Small CNN/MLP scaffold with switchable shape prediction.

    ``forward(height_m, valid_mask, action)`` is identical in both modes.  The
    amount-only mode does not instantiate the spatial outcome head, preventing
    accidental shape supervision from leaking into that baseline.
    """

    def __init__(
        self,
        config: PredictorConfig | None = None,
        contract: ScoopDataContract = DEFAULT_DATA_CONTRACT,
    ) -> None:
        super().__init__()
        self.config = config or PredictorConfig()
        self.mode = self.config.resolved_mode()
        self.contract = contract
        c = self.config.spatial_channels

        # height/fill and validity are deliberately separate channels.
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(2, c, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Conv2d(c, 2 * c, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(4, self.config.action_features),
            nn.SiLU(),
            nn.Linear(self.config.action_features, self.config.action_features),
            nn.SiLU(),
        )
        fused_features = 2 * c + self.config.action_features
        self.scalar_trunk = nn.Sequential(
            nn.Linear(fused_features, self.config.hidden_features),
            nn.SiLU(),
            nn.Linear(self.config.hidden_features, self.config.hidden_features),
            nn.SiLU(),
        )
        # mass raw mean/log-variance, time raw mean/log-variance, failure logit
        self.scalar_head = nn.Linear(self.config.hidden_features, 5)

        if self.mode is PredictionMode.AMOUNT_AND_SHAPE:
            self.shape_action_projection = nn.Linear(4, self.config.action_features)
            self.shape_head = nn.Sequential(
                nn.Conv2d(c + self.config.action_features, 2 * c, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(2 * c, c, 3, padding=1),
                nn.SiLU(),
                # next-height delta raw mean and log-variance
                nn.Conv2d(c, 2, 1),
            )
        else:
            self.shape_action_projection = None
            self.shape_head = None

    def forward(self, height_m: Tensor, valid_mask: Tensor, action: Tensor) -> ScoopPrediction:
        validate_model_inputs(height_m, valid_mask, action, self.contract)

        scaled_height = height_m / self.config.height_scale_m
        observation = torch.cat((scaled_height, valid_mask.to(height_m.dtype)), dim=1)
        spatial_features = self.spatial_encoder(observation)
        global_features = self.global_encoder(spatial_features)
        action_features = self.action_encoder(action)
        scalar_raw = self.scalar_head(
            self.scalar_trunk(torch.cat((global_features, action_features), dim=1))
        )

        mass_mean = F.softplus(scalar_raw[:, 0:1]) * self.config.mass_scale_kg
        mass_variance = (
            F.softplus(scalar_raw[:, 1:2]) * self.config.mass_scale_kg**2
            + self.config.min_mass_variance_kg2
        )
        time_mean = F.softplus(scalar_raw[:, 2:3]) * self.config.time_scale_s
        time_variance = (
            F.softplus(scalar_raw[:, 3:4]) * self.config.time_scale_s**2
            + self.config.min_time_variance_s2
        )
        failure_probability = torch.sigmoid(scalar_raw[:, 4:5])
        failure_variance = failure_probability * (1.0 - failure_probability)

        next_height_mean: Tensor | None = None
        next_height_variance: Tensor | None = None
        if self.mode is PredictionMode.AMOUNT_AND_SHAPE:
            assert self.shape_action_projection is not None
            assert self.shape_head is not None
            shape_action = self.shape_action_projection(action)
            shape_action = shape_action[:, :, None, None].expand(
                -1, -1, spatial_features.shape[2], spatial_features.shape[3]
            )
            shape_raw = self.shape_head(torch.cat((spatial_features, shape_action), dim=1))
            delta_mean_m = (
                torch.tanh(shape_raw[:, 0:1]) * self.config.height_delta_limit_m
            )
            next_height_mean = torch.clamp_min(height_m + delta_mean_m, 0.0)
            next_height_variance = (
                F.softplus(shape_raw[:, 1:2]) * self.config.height_delta_limit_m**2
                + self.config.min_height_variance_m2
            )

        return ScoopPrediction(
            mode=self.mode,
            scooped_mass_mean_kg=mass_mean,
            scooped_mass_variance_kg2=mass_variance,
            elapsed_time_mean_s=time_mean,
            elapsed_time_variance_s2=time_variance,
            failure_probability=failure_probability,
            failure_variance=failure_variance,
            next_height_mean_m=next_height_mean,
            next_height_variance_m2=next_height_variance,
        )


@dataclass(frozen=True)
class DecisionRiskConfig:
    """Explicit, validation-tuned coefficients for candidate action ranking.

    No coefficient is presented as a measured project constant.  A caller must
    choose them on held-out rollout data before using the score in an
    experiment.
    """

    confidence_std: float
    failure_penalty_s: float
    shape_std_penalty_s_per_m: float
    minimum_denominator_s: float = 1.0e-6

    def __post_init__(self) -> None:
        for name, value in (
            ("confidence_std", self.confidence_std),
            ("failure_penalty_s", self.failure_penalty_s),
            ("shape_std_penalty_s_per_m", self.shape_std_penalty_s_per_m),
            ("minimum_denominator_s", self.minimum_denominator_s),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_denominator_s == 0:
            raise ValueError("minimum_denominator_s must be > 0")


def risk_adjusted_candidate_score(
    prediction: ScoopPrediction,
    risk: DecisionRiskConfig,
) -> Tensor:
    """Return one uncertainty-aware progress score per candidate.

    Larger is better.  Mass uses a lower confidence bound; time and failure use
    upper confidence bounds.  In shape mode, mean per-cell height standard
    deviation adds a time-equivalent penalty.  A future multi-step planner can
    additionally roll ``next_height_mean_m`` forward as the successor state,
    while retaining this variance penalty/gate.

    The score has units kg/s.  It is a decision-layer scaffold, not a validated
    policy; the three risk coefficients are deliberately mandatory.
    """

    beta = risk.confidence_std
    mass_std = torch.sqrt(torch.clamp_min(prediction.scooped_mass_variance_kg2, 0.0))
    time_std = torch.sqrt(torch.clamp_min(prediction.elapsed_time_variance_s2, 0.0))
    failure_std = torch.sqrt(torch.clamp_min(prediction.failure_variance, 0.0))

    mass_lcb = torch.clamp_min(prediction.scooped_mass_mean_kg - beta * mass_std, 0.0)
    time_ucb = torch.clamp_min(prediction.elapsed_time_mean_s + beta * time_std, 0.0)
    failure_ucb = torch.clamp(
        prediction.failure_probability + beta * failure_std, 0.0, 1.0
    )
    denominator_s = time_ucb + risk.failure_penalty_s * failure_ucb

    if prediction.mode is PredictionMode.AMOUNT_AND_SHAPE:
        if prediction.next_height_variance_m2 is None:
            raise ValueError("shape mode prediction is missing height variance")
        height_std_m = torch.sqrt(
            torch.clamp_min(prediction.next_height_variance_m2, 0.0)
        )
        mean_height_std_m = height_std_m.mean(dim=(1, 2, 3), keepdim=False)[:, None]
        denominator_s = (
            denominator_s
            + risk.shape_std_penalty_s_per_m * mean_height_std_m
        )

    return mass_lcb / torch.clamp_min(denominator_s, risk.minimum_denominator_s)


def select_risk_adjusted_candidate(
    prediction: ScoopPrediction,
    risk: DecisionRiskConfig,
) -> tuple[int, Tensor]:
    """Choose the highest-scoring row from a candidate batch."""

    score = risk_adjusted_candidate_score(prediction, risk)
    if score.ndim != 2 or score.shape[1] != 1 or score.shape[0] == 0:
        raise ValueError(f"candidate score must be non-empty [N, 1], got {score.shape}")
    return int(torch.argmax(score[:, 0]).item()), score


@dataclass(frozen=True)
class ScoopEpisodeMetrics:
    """Primary evaluation axes plus the explicitly secondary distance axis."""

    total_time_s: float
    total_scoops: int
    failure_rate: float
    total_distance_m: float | None = None


def _to_1d_tensor(values: Tensor | Sequence[float] | Iterable[float], name: str) -> Tensor:
    if isinstance(values, Tensor):
        tensor = values.detach().cpu()
    else:
        tensor = torch.as_tensor(list(values))
    if tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be [N] or [N, 1], got {tensor.shape}")
    return tensor


def compute_episode_metrics(
    elapsed_time_s: Tensor | Sequence[float] | Iterable[float],
    failed: Tensor | Sequence[bool] | Iterable[bool],
    travel_distance_m: Tensor | Sequence[float] | Iterable[float] | None = None,
) -> ScoopEpisodeMetrics:
    """Compute total time, scoop count, failure rate, and optional distance."""

    times = _to_1d_tensor(elapsed_time_s, "elapsed_time_s").to(torch.float64)
    failures = _to_1d_tensor(failed, "failed")
    if times.numel() == 0:
        raise ValueError("an episode must contain at least one scoop attempt")
    if failures.numel() != times.numel():
        raise ValueError("failed and elapsed_time_s must have the same length")
    if not torch.isfinite(times).all() or (times < 0).any():
        raise ValueError("elapsed_time_s must be finite and non-negative")
    if failures.dtype != torch.bool:
        if not torch.all((failures == 0) | (failures == 1)):
            raise ValueError("failed must contain bool or 0/1 values")
        failures = failures.to(torch.bool)

    total_distance: float | None = None
    if travel_distance_m is not None:
        distances = _to_1d_tensor(travel_distance_m, "travel_distance_m").to(torch.float64)
        if distances.numel() != times.numel():
            raise ValueError("travel_distance_m and elapsed_time_s must match in length")
        if not torch.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("travel_distance_m must be finite and non-negative")
        total_distance = float(distances.sum().item())

    return ScoopEpisodeMetrics(
        total_time_s=float(times.sum().item()),
        total_scoops=int(times.numel()),
        failure_rate=float(failures.to(torch.float64).mean().item()),
        total_distance_m=total_distance,
    )


def _assert_prediction_contract(
    prediction: ScoopPrediction,
    batch: int,
    contract: ScoopDataContract,
) -> None:
    scalar_shape = (batch, 1)
    for name, value in (
        ("scooped_mass_mean_kg", prediction.scooped_mass_mean_kg),
        ("scooped_mass_variance_kg2", prediction.scooped_mass_variance_kg2),
        ("elapsed_time_mean_s", prediction.elapsed_time_mean_s),
        ("elapsed_time_variance_s2", prediction.elapsed_time_variance_s2),
        ("failure_probability", prediction.failure_probability),
        ("failure_variance", prediction.failure_variance),
    ):
        _require_shape(name, value, scalar_shape)
        _require_float32(name, value)
        assert torch.isfinite(value).all()
    assert (prediction.scooped_mass_variance_kg2 > 0).all()
    assert (prediction.elapsed_time_variance_s2 > 0).all()
    assert (prediction.failure_variance >= 0).all()
    assert ((prediction.failure_probability >= 0) & (prediction.failure_probability <= 1)).all()

    if prediction.mode is PredictionMode.AMOUNT_ONLY:
        assert prediction.next_height_mean_m is None
        assert prediction.next_height_variance_m2 is None
    else:
        assert prediction.next_height_mean_m is not None
        assert prediction.next_height_variance_m2 is not None
        rows, cols = contract.heightmap.shape
        map_shape = (batch, 1, rows, cols)
        _require_shape("next_height_mean_m", prediction.next_height_mean_m, map_shape)
        _require_shape("next_height_variance_m2", prediction.next_height_variance_m2, map_shape)
        _require_float32("next_height_mean_m", prediction.next_height_mean_m)
        _require_float32("next_height_variance_m2", prediction.next_height_variance_m2)
        assert torch.isfinite(prediction.next_height_mean_m).all()
        assert (prediction.next_height_mean_m >= 0).all()
        assert (prediction.next_height_variance_m2 > 0).all()


def run_dummy_forward_check() -> None:
    """CPU-only shape/variance/interface smoke check; performs no optimization."""

    torch.manual_seed(457)
    contract = DEFAULT_DATA_CONTRACT
    batch = 3
    rows, cols = contract.heightmap.shape
    height_m = torch.rand(batch, 1, rows, cols, dtype=torch.float32) * 0.06
    valid_mask = torch.ones(batch, 1, rows, cols, dtype=torch.bool)
    valid_mask[1, :, :2, :3] = False
    height_m[~valid_mask] = 0.0
    action = torch.tensor(
        [
            [0.175, -0.025, 1.0, 0.0],
            [0.200, 0.000, 0.0, 1.0],
            [0.225, 0.025, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    risk = DecisionRiskConfig(
        confidence_std=1.0,
        failure_penalty_s=2.0,
        shape_std_penalty_s_per_m=20.0,
    )

    for mode in PredictionMode:
        model = ScoopPredictor(PredictorConfig(mode=mode), contract).eval()
        with torch.no_grad():
            prediction = model(height_m, valid_mask, action)
            _assert_prediction_contract(prediction, batch, contract)
            selected, score = select_risk_adjusted_candidate(prediction, risk)
        assert 0 <= selected < batch
        assert tuple(score.shape) == (batch, 1)
        assert torch.isfinite(score).all()

        target = ScoopTarget(
            scooped_mass_kg=torch.full((batch, 1), 0.010, dtype=torch.float32),
            elapsed_time_s=torch.full((batch, 1), 3.0, dtype=torch.float32),
            failed=torch.zeros((batch, 1), dtype=torch.bool),
            next_height_m=(height_m.clone() if mode is PredictionMode.AMOUNT_AND_SHAPE else None),
            next_valid_mask=(valid_mask.clone() if mode is PredictionMode.AMOUNT_AND_SHAPE else None),
        )
        validate_targets(target, mode, contract)
        print(
            f"{mode.value}: scalar=(3, 1), "
            f"next_height={'(3, 1, 76, 38)' if prediction.next_height_mean_m is not None else 'None'}, "
            f"variance=PASS, selected={selected}"
        )

    metrics = compute_episode_metrics(
        elapsed_time_s=[3.0, 3.2, 2.8],
        failed=[False, True, False],
        travel_distance_m=[0.4, 0.4, 0.4],
    )
    assert abs(metrics.total_time_s - 9.0) < 1.0e-6
    assert metrics.total_scoops == 3
    assert abs(metrics.failure_rate - 1.0 / 3.0) < 1.0e-12
    assert metrics.total_distance_m is not None
    assert abs(metrics.total_distance_m - 1.2) < 1.0e-6
    print(f"metrics={metrics}")


if __name__ == "__main__":
    run_dummy_forward_check()
