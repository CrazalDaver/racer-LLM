"""Generate a local-model raceline with the TUM minimum-curvature pipeline.

Stages
------
1. Resample the registered project centerline at the physical simulation scale.
2. Optimize lateral offsets with TUM's shortest-path/minimum-curvature solver.
3. Build a closed speed profile from local friction, throttle, and braking limits.
4. Validate and export the CSV layout consumed by :class:`track.RaceLine`.

The full TUM minimum-time optimizer is deliberately not used: its double-track
and tire models do not match this project's kinematic vehicle.  Path geometry
comes from ``trajectory_planning_helpers``; speed limits come from the shared
local vehicle model.

Examples
--------
    # Recommended Berlin candidate and diagnostic plot
    .venv/bin/python train/raceline/optimize_raceline.py --track berlin

    # Geometry ablation
    .venv/bin/python train/raceline/optimize_raceline.py \
        --track berlin --objective shortest

    # Add an explicit 2 cm center-point boundary margin
    .venv/bin/python train/raceline/optimize_raceline.py \
        --track berlin --boundary-margin 0.02

Important arguments
-------------------
``--track``: berlin, modena, or rounded_rectangle.
``--objective``: mincurv_iqp (recommended), mincurv, or shortest.
``--opt-step``: approximate centerline spacing used by the geometry QP [m].
``--output-step``: exported raceline spacing at physical scale [m].
``--boundary-margin``: distance kept between the optimized center point and
the configured geometric track boundary [m].
``--output``: optional CSV destination. Spatial columns are stored at the
track's native scale because the simulator applies ``TrackSpec.scale`` later.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

# Allow direct execution from the repository root without installing racer-LLM.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import GRAVITY, make_vehicle_params, params as sim_params
from track import Track, get_track_spec, load_centerline_csv, load_raceline_csv
from vehicle_model import longitudinal_acceleration, steering_angle

try:
    import trajectory_planning_helpers as tph
except ImportError as exc:  # pragma: no cover - exercised only on missing optional dependency
    raise SystemExit(
        "Missing raceline dependency. Run:\n"
        "  .venv/bin/pip install trajectory-planning-helpers\n"
        "On macOS/ARM, replace its broken legacy QP wheel with:\n"
        "  .venv/bin/pip install --upgrade --force-reinstall quadprog==0.1.13"
    ) from exc


OBJECTIVES = ("mincurv_iqp", "mincurv", "shortest")


def _periodic_path_indices(points: np.ndarray, tolerance: float = 1.0e-9) -> np.ndarray:
    """Return row indices for one clean period of a closed path."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("path points must have shape (N, 2)")
    if points.shape[0] < 3:
        raise ValueError("a closed path requires at least three points")
    endpoint = points.shape[0]
    if np.linalg.norm(points[0] - points[-1]) <= tolerance:
        endpoint -= 1
    indices = np.arange(endpoint)
    candidates = points[:endpoint]
    keep = np.r_[True, np.linalg.norm(np.diff(candidates, axis=0), axis=1) > tolerance]
    indices = indices[keep]
    if indices.size < 3:
        raise ValueError("a closed path requires at least three distinct points")
    return indices


def _periodic_path_points(points: np.ndarray, tolerance: float = 1.0e-9) -> np.ndarray:
    """Return one clean period of a closed path.

    Some released alpha-RACER CSVs explicitly repeat their first waypoint as
    the final row, while locally generated files do not.  Remove that duplicate
    endpoint, as well as any zero-length consecutive segments, before building
    a strictly increasing arc-length coordinate for periodic splines.
    """
    points = np.asarray(points, dtype=float)
    return points[_periodic_path_indices(points, tolerance)]


def _closed_arc_length(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = _periodic_path_points(points)
    closed = np.vstack((points, points[0]))
    lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    return np.r_[0.0, np.cumsum(lengths)], lengths


def _resample_closed_centerline(points: np.ndarray, step: float) -> np.ndarray:
    """Return a periodic, approximately equidistant centerline."""
    points = _periodic_path_points(points)
    s, _ = _closed_arc_length(points)
    closed = np.vstack((points, points[0]))
    spline = CubicSpline(s, closed, axis=0, bc_type="periodic")
    count = max(12, int(math.ceil(float(s[-1]) / float(step))))
    return np.asarray(
        spline(np.linspace(0.0, float(s[-1]), count, endpoint=False)),
        dtype=float,
    )


def _prepare_reference_track(
    centerline: np.ndarray,
    half_width: float,
    boundary_margin: float,
    opt_step: float,
):
    usable_half_width = float(half_width) - float(boundary_margin)
    if usable_half_width <= 0.0:
        raise ValueError("boundary margin must be smaller than the track half-width")
    reference = _resample_closed_centerline(centerline, opt_step)
    reftrack = np.column_stack((
        reference,
        np.full(reference.shape[0], usable_half_width),
        np.full(reference.shape[0], usable_half_width),
    ))
    closed_reference = np.vstack((reference, reference[0]))
    coeffs_x, coeffs_y, matrix_a, normals = tph.calc_splines.calc_splines(
        path=closed_reference,
        use_dist_scaling=False,
    )
    if tph.check_normals_crossing.check_normals_crossing(
        track=reftrack,
        normvec_normalized=normals,
        horizon=10,
    ):
        raise RuntimeError("reference-track spline normals cross; increase --opt-step")
    return reftrack, normals, matrix_a, coeffs_x, coeffs_y


def _optimize_offsets(
    objective: str,
    reftrack: np.ndarray,
    normals: np.ndarray,
    matrix_a: np.ndarray,
    coeffs_x: np.ndarray,
    coeffs_y: np.ndarray,
    curvature_bound: float,
    opt_step: float,
    iqp_iterations: int,
    iqp_curvature_error: float,
):
    diagnostics = {"curvature_linearization_error_1pm": None}
    if objective == "shortest":
        alpha = tph.opt_shortest_path.opt_shortest_path(
            reftrack=reftrack,
            normvectors=normals,
            w_veh=0.0,
            print_debug=False,
        )
        return alpha, reftrack, normals, diagnostics

    if objective == "mincurv":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module=r"trajectory_planning_helpers\.opt_min_curv",
            )
            alpha, curvature_error = tph.opt_min_curv.opt_min_curv(
                reftrack=reftrack,
                normvectors=normals,
                A=matrix_a,
                kappa_bound=curvature_bound,
                w_veh=0.0,
                print_debug=False,
                plot_debug=False,
            )
        diagnostics["curvature_linearization_error_1pm"] = float(curvature_error)
        if curvature_error > iqp_curvature_error:
            warnings.warn(
                f"single-QP curvature linearization error is {curvature_error:.3f} 1/m; "
                "the exported path will be checked using its actual spline curvature",
                RuntimeWarning,
            )
        return alpha, reftrack, normals, diagnostics

    spline_lengths = tph.calc_spline_lengths.calc_spline_lengths(
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
    )
    indices = np.arange(reftrack.shape[0])
    zeros = np.zeros(reftrack.shape[0])
    heading, curvature, curvature_rate = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        ind_spls=indices,
        t_spls=zeros,
        calc_curv=True,
        calc_dcurv=True,
    )
    # TPH 0.79 emits benign NumPy-2 overflow warnings while forming its first
    # linearized matrices on small-scale tracks. The result is accepted only
    # after the explicit finite/boundary/curvature checks below.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"trajectory_planning_helpers\.opt_min_curv",
        )
        result = tph.iqp_handler.iqp_handler(
            reftrack=reftrack,
            normvectors=normals,
            A=matrix_a,
            spline_len=spline_lengths,
            psi=heading,
            kappa=curvature,
            dkappa=curvature_rate,
            kappa_bound=curvature_bound,
            w_veh=0.0,
            print_debug=False,
            plot_debug=False,
            stepsize_interp=opt_step,
            iters_min=iqp_iterations,
            curv_error_allowed=iqp_curvature_error,
        )
    alpha, final_reftrack, final_normals = result[:3]
    return alpha, final_reftrack, final_normals, diagnostics


def _create_raceline(
    reftrack: np.ndarray,
    normals: np.ndarray,
    alpha: np.ndarray,
    output_step: float,
):
    result = tph.create_raceline.create_raceline(
        refline=reftrack[:, :2],
        normvectors=normals,
        alpha=alpha,
        stepsize_interp=output_step,
    )
    (
        raceline,
        _,
        coeffs_x,
        coeffs_y,
        spline_indices,
        spline_t,
        progress,
        _,
        element_lengths,
    ) = result
    heading, curvature = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        ind_spls=spline_indices,
        t_spls=spline_t,
    )
    return raceline, progress, element_lengths, heading, curvature


def _local_speed_profile(
    curvature: np.ndarray,
    element_lengths: np.ndarray,
    *,
    vehicle_params,
    friction_coefficient: float,
    max_speed: float,
    max_iterations: int = 100,
    tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Closed profile using the same limits as the local alpha-RACER MPC.

    The released controller constrains the normalized steering command rather
    than geometric curvature directly:

        v^2 |tan(u_delta)| <= mu g L.

    Convert each path curvature to the command required by the shared
    kinematic steering map before constructing the lateral speed limit.
    """
    wheelbase = float(vehicle_params.LF + vehicle_params.LR)
    curvature = np.asarray(curvature, dtype=float)
    steering_angle_required = np.arctan(wheelbase * curvature)
    steering_command_required = (
        steering_angle_required - float(vehicle_params.Sb)
    ) / float(vehicle_params.Sa)
    lateral_proxy = np.maximum(np.abs(np.tan(steering_command_required)), 1.0e-9)
    lateral_limit = float(friction_coefficient) * GRAVITY * wheelbase
    speed = np.minimum(float(max_speed), np.sqrt(lateral_limit / lateral_proxy))
    speed = np.maximum(speed, 0.05)
    ds = np.asarray(element_lengths, dtype=float)

    physical_acceleration_limit = float(friction_coefficient) * GRAVITY

    def available_longitudinal(v: float, kappa: float) -> float:
        lateral_accel = min(v * v * abs(kappa), physical_acceleration_limit)
        return math.sqrt(max(
            physical_acceleration_limit * physical_acceleration_limit
            - lateral_accel * lateral_accel,
            0.0,
        ))

    for _ in range(max_iterations):
        previous = speed.copy()
        for idx in range(speed.size):
            nxt = (idx + 1) % speed.size
            drive = max(
                float(longitudinal_acceleration(1.0, speed[idx], vehicle_params)),
                0.0,
            )
            accel = min(drive, available_longitudinal(speed[idx], curvature[idx]))
            reachable = math.sqrt(max(speed[idx] ** 2 + 2.0 * accel * ds[idx], 0.0))
            speed[nxt] = min(speed[nxt], reachable, max_speed)

        for idx in range(speed.size - 1, -1, -1):
            nxt = (idx + 1) % speed.size
            braking = max(
                -float(longitudinal_acceleration(-1.0, speed[nxt], vehicle_params)),
                0.0,
            )
            decel = min(braking, available_longitudinal(speed[idx], curvature[idx]))
            reachable = math.sqrt(max(speed[nxt] ** 2 + 2.0 * decel * ds[idx], 0.0))
            speed[idx] = min(speed[idx], reachable, max_speed)

        if float(np.max(np.abs(speed - previous))) <= tolerance:
            break
    return speed


def _lap_time(speed: np.ndarray, element_lengths: np.ndarray) -> float:
    next_speed = np.roll(speed, -1)
    segment_speed = np.maximum(0.5 * (speed + next_speed), 1.0e-6)
    return float(np.sum(element_lengths / segment_speed))


def _centerline_deviation(centerline: np.ndarray, raceline: np.ndarray) -> np.ndarray:
    """Signed shortest distance to the closed centerline polyline.

    The runtime projector is intentionally scalar and therefore slow for a
    dense offline path.  A KD-tree selects nearby segment midpoints here; the
    final projection and signed distance are still computed exactly on those
    line segments.
    """
    centerline = np.asarray(centerline, dtype=float)
    raceline = np.asarray(raceline, dtype=float)
    p0 = centerline
    p1 = np.roll(centerline, -1, axis=0)
    segment = p1 - p0
    length_sq = np.maximum(np.sum(segment * segment, axis=1), 1.0e-12)
    midpoints = 0.5 * (p0 + p1)
    candidate_count = min(12, centerline.shape[0])
    _, candidates = cKDTree(midpoints).query(raceline, k=candidate_count)
    if candidates.ndim == 1:
        candidates = candidates[:, None]

    candidate_p0 = p0[candidates]
    candidate_segments = segment[candidates]
    delta = raceline[:, None, :] - candidate_p0
    fraction = np.clip(
        np.sum(delta * candidate_segments, axis=2) / length_sq[candidates],
        0.0,
        1.0,
    )
    projected = candidate_p0 + fraction[..., None] * candidate_segments
    residual = raceline[:, None, :] - projected
    distance_sq = np.sum(residual * residual, axis=2)
    best = np.argmin(distance_sq, axis=1)
    rows = np.arange(raceline.shape[0])
    best_residual = residual[rows, best]
    best_segment = candidate_segments[rows, best]
    best_norm = np.maximum(np.linalg.norm(best_segment, axis=1), 1.0e-12)
    left_normal = np.column_stack((-best_segment[:, 1], best_segment[:, 0])) / best_norm[:, None]
    return np.sum(best_residual * left_normal, axis=1)


def _path_curvature(points: np.ndarray) -> np.ndarray:
    points = _periodic_path_points(points)
    s, _ = _closed_arc_length(points)
    closed = np.vstack((points, points[0]))
    spline = CubicSpline(s, closed, axis=0, bc_type="periodic")
    query = s[:-1]
    first = spline(query, 1)
    second = spline(query, 2)
    denominator = np.maximum(np.sum(first * first, axis=1) ** 1.5, 1.0e-12)
    return (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / denominator


def _existing_raceline_metrics(path: str, scale: float) -> dict:
    existing = load_raceline_csv(path, scale=scale)
    clean_indices = _periodic_path_indices(existing.waypoints)
    waypoints = existing.waypoints[clean_indices]
    speed = existing.speed[clean_indices]
    _, lengths = _closed_arc_length(waypoints)
    curvature = _path_curvature(waypoints)
    return {
        "path": path,
        "length_m": float(np.sum(lengths)),
        "estimated_lap_time_s": _lap_time(speed, lengths),
        "mean_speed_mps": float(np.sum(lengths) / _lap_time(speed, lengths)),
        "max_abs_curvature_1pm": float(np.max(np.abs(curvature))),
        "min_speed_mps": float(np.min(speed)),
        "max_speed_mps": float(np.max(speed)),
    }


def _save_plot(
    output_path: Path,
    centerline: np.ndarray,
    half_width: float,
    raceline: np.ndarray,
    progress: np.ndarray,
    speed: np.ndarray,
    curvature: np.ndarray,
):
    track = Track(centerline)
    sample_s = np.linspace(0.0, track.total_length, 1000, endpoint=False)
    center = np.asarray([track.interpolate(s)[:2] for s in sample_s])
    heading = np.asarray([track.interpolate(s)[2] for s in sample_s])
    normals = np.column_stack((-np.sin(heading), np.cos(heading)))
    left = center + half_width * normals
    right = center - half_width * normals

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(left[:, 0], left[:, 1], color="0.7", linewidth=1)
    axes[0].plot(right[:, 0], right[:, 1], color="0.7", linewidth=1)
    axes[0].plot(center[:, 0], center[:, 1], "--", color="0.5", label="centerline")
    axes[0].plot(raceline[:, 0], raceline[:, 1], color="tab:red", label="optimized")
    axes[0].axis("equal")
    axes[0].set_title("Local optimized raceline")
    axes[0].legend()

    axes[1].plot(progress, speed, label="speed [m/s]")
    axes[1].plot(progress, np.abs(curvature), label="|curvature| [1/m]")
    axes[1].set_xlabel("raceline progress [m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def optimize(args) -> tuple[Path, dict]:
    spec = get_track_spec(args.track)
    centerline = load_centerline_csv(spec.centerline, scale=spec.scale)
    half_width = spec.half_width
    vehicle_params = make_vehicle_params(args.vehicle_model)
    wheelbase = vehicle_params.LF + vehicle_params.LR
    maximum_angle = max(
        abs(float(steering_angle(-1.0, vehicle_params))),
        abs(float(steering_angle(1.0, vehicle_params))),
    )
    physical_curvature_bound = abs(math.tan(maximum_angle)) / wheelbase
    curvature_bound = (
        physical_curvature_bound
        if args.curvature_bound is None
        else float(args.curvature_bound)
    )

    # The QP constrains lateral offsets at reference samples, while the final
    # periodic spline can bow slightly between them. Reserve half an output
    # sample as a numerical interpolation buffer, then validate the requested
    # physical margin independently below.
    interpolation_buffer = 0.5 * args.output_step
    reftrack, normals, matrix_a, coeffs_x, coeffs_y = _prepare_reference_track(
        centerline,
        half_width,
        args.boundary_margin + interpolation_buffer,
        args.opt_step,
    )
    alpha, final_reftrack, final_normals, optimizer_diagnostics = _optimize_offsets(
        args.objective,
        reftrack,
        normals,
        matrix_a,
        coeffs_x,
        coeffs_y,
        curvature_bound,
        args.opt_step,
        args.iqp_iterations,
        args.iqp_curvature_error,
    )
    raceline, progress, element_lengths, heading, curvature = _create_raceline(
        final_reftrack,
        final_normals,
        alpha,
        args.output_step,
    )
    if not all(np.all(np.isfinite(array)) for array in (
        raceline, progress, element_lengths, heading, curvature
    )):
        raise RuntimeError("optimizer produced non-finite trajectory data")

    deviations = _centerline_deviation(centerline, raceline)
    allowed_offset = half_width - args.boundary_margin
    max_offset = float(np.max(np.abs(deviations)))
    boundary_validation_tolerance = 5.0e-3
    if max_offset > allowed_offset + boundary_validation_tolerance:
        raise RuntimeError(
            f"optimized path exceeds usable half-width: {max_offset:.4f} > {allowed_offset:.4f} m"
        )
    max_curvature = float(np.max(np.abs(curvature)))
    if max_curvature > curvature_bound + 5.0e-2:
        raise RuntimeError(
            f"optimized curvature exceeds limit: {max_curvature:.4f} > {curvature_bound:.4f} 1/m"
        )

    speed = _local_speed_profile(
        curvature,
        element_lengths,
        vehicle_params=vehicle_params,
        friction_coefficient=args.friction_coefficient,
        max_speed=args.max_speed,
    )
    lap_time = _lap_time(speed, element_lengths)
    length = float(np.sum(element_lengths))

    default_output = (
        REPO_ROOT
        / "track_csv"
        / "generated"
        / f"{args.track}_{args.objective}_local.csv"
    )
    output_path = default_output if args.output is None else Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # TrackSpec applies scale again at load time, so spatial values are stored
    # in the native CSV scale while velocities remain physical m/s.
    exported = np.column_stack((
        progress / spec.scale,
        raceline[:, 0] / spec.scale,
        raceline[:, 1] / spec.scale,
        speed,
        deviations / spec.scale,
    ))
    np.savetxt(
        output_path,
        exported,
        delimiter=",",
        header="s_m,x_m,y_m,v_mps,deviation_m",
        comments="# ",
        fmt="%.9f",
    )

    metadata = {
        "source": "TUMFTM/global_racetrajectory_optimization geometry pipeline",
        "trajectory_planning_helpers_version": importlib.metadata.version(
            "trajectory-planning-helpers"
        ),
        "track": args.track,
        "source_centerline": spec.centerline,
        "previous_raceline": _existing_raceline_metrics(spec.raceline, spec.scale),
        "objective": args.objective,
        "vehicle_model": args.vehicle_model,
        "track_scale": spec.scale,
        "track_half_width_m": half_width,
        "boundary_margin_m": args.boundary_margin,
        "spline_interpolation_buffer_m": interpolation_buffer,
        "boundary_validation_tolerance_m": boundary_validation_tolerance,
        "friction_coefficient": args.friction_coefficient,
        "max_speed_mps": args.max_speed,
        "curvature_bound_1pm": curvature_bound,
        "opt_step_m": args.opt_step,
        "output_step_m": args.output_step,
        "point_count": int(raceline.shape[0]),
        "length_m": length,
        "estimated_lap_time_s": lap_time,
        "estimated_mean_speed_mps": length / lap_time,
        "min_speed_mps": float(np.min(speed)),
        "max_speed_mps": float(np.max(speed)),
        "mean_abs_curvature_1pm": float(np.mean(np.abs(curvature))),
        "max_abs_curvature_1pm": max_curvature,
        "max_abs_centerline_deviation_m": max_offset,
        "output_csv": str(output_path.relative_to(REPO_ROOT)),
        **optimizer_diagnostics,
    }
    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    _save_plot(
        output_path,
        centerline,
        half_width,
        raceline,
        progress,
        speed,
        curvature,
    )
    return output_path, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a project-compatible local raceline and speed profile."
    )
    parser.add_argument("--track", choices=("berlin", "modena", "rounded_rectangle"), default="berlin")
    parser.add_argument("--objective", choices=OBJECTIVES, default="mincurv_iqp")
    parser.add_argument("--vehicle-model", choices=("kinematic", "dbm"), default="kinematic")
    parser.add_argument("--opt-step", type=float, default=0.2, metavar="METERS")
    parser.add_argument("--output-step", type=float, default=0.04, metavar="METERS")
    parser.add_argument("--boundary-margin", type=float, default=0.05, metavar="METERS")
    parser.add_argument("--curvature-bound", type=float, default=None, metavar="1/METERS")
    parser.add_argument("--friction-coefficient", type=float, default=sim_params.MPC_FRICTION_COEFFICIENT)
    parser.add_argument("--max-speed", type=float, default=sim_params.MAX_STATE_SPEED, metavar="MPS")
    parser.add_argument("--iqp-iterations", type=int, default=5)
    parser.add_argument("--iqp-curvature-error", type=float, default=0.08, metavar="1/METERS")
    parser.add_argument("--output", default=None, metavar="CSV")
    args = parser.parse_args()
    if args.opt_step <= 0.0 or args.output_step <= 0.0:
        parser.error("--opt-step and --output-step must be positive")
    if args.boundary_margin < 0.0:
        parser.error("--boundary-margin must be non-negative")
    if args.friction_coefficient <= 0.0 or args.max_speed <= 0.0:
        parser.error("friction coefficient and maximum speed must be positive")
    if args.iqp_iterations < 1 or args.iqp_curvature_error <= 0.0:
        parser.error("IQP iterations and curvature error must be positive")
    return args


if __name__ == "__main__":
    output, metrics = optimize(parse_args())
    previous = metrics["previous_raceline"]
    print(f"Saved raceline: {output.relative_to(REPO_ROOT)}")
    print(
        "New: "
        f"length={metrics['length_m']:.2f} m, "
        f"estimated lap={metrics['estimated_lap_time_s']:.2f} s, "
        f"mean speed={metrics['estimated_mean_speed_mps']:.2f} m/s, "
        f"max |kappa|={metrics['max_abs_curvature_1pm']:.3f} 1/m"
    )
    print(
        "Previous CSV: "
        f"length={previous['length_m']:.2f} m, "
        f"estimated lap={previous['estimated_lap_time_s']:.2f} s, "
        f"mean speed={previous['mean_speed_mps']:.2f} m/s, "
        f"max |kappa|={previous['max_abs_curvature_1pm']:.3f} 1/m"
    )
