"""Offline optimal raceline optimization and GGV speed planning.

Source: based on alpha-RACER repository
            - ref_trajs/calculate_raceline.py and
            - global_racetrajectory_optimization/main_globaltraj.py

This this the script for optimal raceline generation from 
a smooth track centerline points csv.

Main functions:
    - prepare_track: smooth and resample the reference centerline.
    - optimize: smooth track centerline points reference -> chosen optimizing objective -> GGV speed.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import splev, splprep
from scipy.optimize import fmin
import trajectory_planning_helpers as tph

from config import TRACK_DIR, TRACKS
from .track import load_centerline_csv, load_raceline_csv

# min curvature as default
OPT_TYPES = ("mincurv", "mincurv_iqp", "shortest_path")


def prepare_track(track):
    """
    Smooth and resample the reference centerline, compensating track widths.
    Preprocess the track for subsequent QP-based raceline optimization.
    Based on TPH's spline_approximation, as used by alpha-RACER.    
    
    Interpolate -> Spline -> Resample -> Compute left/right boundaries

    Input:
        track: (N, 4) array [x, y, right_width, left_width] in source CSV units.

    Output:
        (M, 4) array with the same columns and units. 
        centerline points sampled from a smooth periodic spline; 
        widths are adjusted for the reference line's displacement. 
    """

    # Interpolate the track to a uniform step size
    interpolated = tph.interp_track.interp_track(track=track, stepsize=0.03)
    interpolated = np.vstack((interpolated, interpolated[0]))
    # Use a spline to smooth the track and compute
    tck = splprep(interpolated[:, :2].T, k=3, s=10, per=1)[0]
    closed = np.vstack((track, track[0]))  # ensure closeness
    chord_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(closed[:, :2], axis=0), axis=1))]  # compute s(progress) for original closed polyline
    # Resample the spline at a uniform step size along the chord length
    length_samples = np.asarray(splev(np.linspace(0, 1, int(np.ceil(chord_s[-1])) * 4), tck)).T
    length = np.sum(np.linalg.norm(np.diff(length_samples, axis=0), axis=1))
    count = int(np.ceil(length / 0.03)) + 1
    parameters = np.linspace(0, 1, count)
    center = np.asarray(splev(parameters, tck)).T[:-1]  # remove duplicate last point
    closest_t = np.empty(len(closed))
    closest = np.empty((len(closed), 2))
    # Compute left and right boundaries based on the closest points on the spline and track widths
    for i, point in enumerate(closed[:, :2]):
        closest_t[i] = fmin(
            lambda u: np.linalg.norm(point - np.asarray(splev(float(u[0]), tck))),
            x0=chord_s[i] / chord_s[-1], disp=False).item()
        closest[i] = splev(closest_t[i], tck)
    distance = np.linalg.norm(closed[:, :2] - closest, axis=1)
    sides = np.array([tph.side_of_line.side_of_line(
        a=closed[i, :2], b=closed[i + 1, :2], z=closest[i]) for i in range(len(track))])
    sides = np.r_[sides, sides[0]]
    right = np.interp(parameters, closest_t, closed[:, 2] + sides * distance)[:-1]
    left = np.interp(parameters, closest_t, closed[:, 3] - sides * distance)[:-1]
    return np.column_stack((center, right, left))


def centerline_deviation(centerline, raceline):
    """
    Raceline result processing:
    Compute the e (signed deviation) of the raceline from the original centerline.

    Input:
        - centerline: (N, 2) array of original track centerline points (x, y).
        - raceline: (M, 2) array of optimized raceline points (x, y).

    Output:
        - deviations: (M,) array of signed deviations from the centerline.
    """
    p0 = np.asarray(centerline)
    segment = np.roll(p0, -1, axis=0) - p0
    length_sq = np.sum(segment * segment, axis=1)  # squared lengths of each segment
    assert not np.any(length_sq <= 0), "centerline has a zero-length segment"
    normals = np.column_stack((-segment[:, 1], segment[:, 0])) / np.sqrt(length_sq)[:, None]
    deviations = np.empty(len(raceline))
    for start in range(0, len(raceline), 128):
        query = raceline[start:start + 128]
        fraction = np.clip(np.sum((query[:, None] - p0) * segment, axis=2)
                           / length_sq, 0, 1)
        residual = query[:, None] - (p0 + fraction[..., None] * segment)
        best = np.argmin(np.sum(residual * residual, axis=2), axis=1)
        rows = np.arange(len(query))
        deviations[start:start + len(query)] = np.sum(residual[rows, best] * normals[best], axis=1)
    return deviations


def speed_profile(curvature, ds):
    """
    Raceline result processing:
    Compute the GGV speed profile along the raceline based on curvature and segment lengths.

    Input:
        - curvature: (M,) array of curvature values along the raceline.
        - ds: (M,) array of segment lengths along the raceline.

    Output:
        - speed: (M,) array of GGV speed values along the raceline.
    """
    curvature, ds = np.asarray(curvature), np.asarray(ds)
    assert (curvature.shape == ds.shape and np.isfinite(curvature).all()
            and np.isfinite(ds).all() and np.all(ds > 0)), "invalid curvature or segment lengths"
    ggv = np.array([[0., 3., .8 * 9.81], [5., 3., .8 * 9.81]])
    machine = np.array([[0., 3.], [5., 2.5]])
    return tph.calc_vel_profile.calc_vel_profile(
        ggv=ggv, ax_max_machines=machine, v_max=5., kappa=curvature,
        el_lengths=ds, closed=True, filt_window=None, dyn_model_exp=1.,
        drag_coeff=0., m_veh=1.)


def optimize(centerline, half_width, opt_type="mincurv"):
    """
    Smooth track centerline points reference -> chosen geometric optimizer -> GGV speed.

    Input:
        - centerline: Nx2 array of source units, smooth track centerline points (x,y).
        - half_width: float, source units as centerline, need scaling.
        - opt_type: string, one of OPT_TYPES
    """
    if opt_type not in OPT_TYPES:
        raise ValueError(f"supported geometric opt_types: {OPT_TYPES}")

    centerline = np.asarray(centerline, dtype=float)
    center = centerline
    # check the centerline closure, remove possible duplicate last point
    if np.linalg.norm(center[0] - center[-1]) < 1e-9:  
        center = center[:-1]
    assert half_width > 0.01 / 2, "vehicle width exceeds the track width"

    # reftrack: [x, y, half_width, half_width]
    reftrack = np.column_stack((center, np.full((len(center), 2), half_width)))
    # Preprocessing: [x, y, half_width, half_width] with smooth centerline
    reftrack = prepare_track(reftrack)
    center = reftrack[:, :2]  # (x,y) of the smoothed centerline
    # compute coeff, matrix and normals for later optimization
    cx, cy, matrix, normals = tph.calc_splines.calc_splines(
        path=np.vstack((center, center[0])))
    if tph.check_normals_crossing.check_normals_crossing(track=reftrack, normvec_normalized=normals, horizon=10):
        raise ValueError("optimizer reference normals cross")

    # Optimization
    bound = 104.0  # Official racecar.ini: width_opt=.01 and curvlim=104 (source units).
    linearization_error = None
    if opt_type == "mincurv":
        offsets, linearization_error = tph.opt_min_curv.opt_min_curv(
            reftrack=reftrack, normvectors=normals, A=matrix,
            kappa_bound=bound, w_veh=0.01,
            print_debug=False, plot_debug=False)
    elif opt_type == "mincurv_iqp":
        # Official racecar.ini: minimum 3 iterations, curvature error .01.
        # IQP updates the reference and its normals; using the old reference
        # after this call would silently reconstruct the wrong path.
        offsets, reftrack, normals = tph.iqp_handler.iqp_handler(
            reftrack=reftrack, normvectors=normals, A=matrix,
            kappa_bound=bound, w_veh=.01, print_debug=False, plot_debug=False,
            stepsize_interp=.03, iters_min=3, curv_error_allowed=.01)
        center = reftrack[:, :2]
    else:
        # Upstream "shortest_path" is a QP on squared chord lengths, not
        # an exact sum of Euclidean lengths and certainly not a lap-time NLP.
        offsets = tph.opt_shortest_path.opt_shortest_path(
            reftrack=reftrack, normvectors=normals, w_veh=.01, print_debug=False)
    # raceline results: points, curvature, speed, and deviation from the smoothed centerline
    result = tph.create_raceline.create_raceline(refline=center,
        normvectors=normals, alpha=offsets, stepsize_interp=0.03)
    points, _, cx, cy, indices, t, progress, _, ds = result
    _, curvature = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=cx, coeffs_y=cy, ind_spls=indices, t_spls=t)
    e = centerline_deviation(centerline, points)
    speed = speed_profile(curvature, ds)
    values = np.column_stack((progress, points, speed, e))
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite optimizer output")
    metrics = {
        "opt_type": opt_type,
        "objective": "squared_segment_lengths" if opt_type == "shortest_path" else "squared_curvature",
        "max_abs_offset_m": float(np.max(np.abs(e))),
        "max_abs_curvature_1pm": float(np.max(np.abs(curvature))),
        "curvature_bound_1pm": float(bound),
        "curvature_excess_1pm": float(max(0, np.max(np.abs(curvature)) - bound)),
        "qp_linearization_error_1pm": None if linearization_error is None else float(linearization_error),
        "length_m": float(np.sum(ds)),
        "estimated_lap_time_s": float(np.sum(2 * ds / (speed + np.roll(speed, -1)))),
        "min_speed_mps": float(np.min(speed)), "max_speed_mps": float(np.max(speed)),
    }
    metrics["minimum_body_clearance_m"] = float(half_width - .01 / 2 - np.max(np.abs(e)))
    # Official check_traj warns about body clearance; it does not turn this
    # approximation into a hard certificate. Keep that warning visible. Never
    # export nonfinite data, a center outside the track or excess configured curvature.
    metrics["exportable"] = bool(metrics["max_abs_offset_m"] <= half_width
                                  and metrics["curvature_excess_1pm"] <= 1e-6)
    return values, curvature, metrics


def main():
    # input arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=TRACKS, default="berlin")
    parser.add_argument("--opt-type", choices=OPT_TYPES, default="mincurv",
                        help="official geometric option; mintime is deferred for separate vehicle adaptation")
    parser.add_argument("--output", type=Path, default=None,
                        help="output stem; default track/<track>_raceline")
    args = parser.parse_args()

    # output path
    suffix = "" if args.opt_type == "mincurv" else f"_{args.opt_type}"
    output = args.output or TRACK_DIR / f"{args.track}{suffix}_raceline"
    output = output.with_suffix(".csv")
    if output.exists():
        parser.error("CSV already exists")
    
    track_config = TRACKS[args.track]
    centerline = load_centerline_csv(track_config.path, scale=1.0)

    # main optimization call
    # Registry widths are physical metres; the optimizer uses source coordinates.
    source_half_width = track_config.half_width_m / track_config.scale
    values, _curvature, metrics = optimize(
        centerline, source_half_width, opt_type=args.opt_type)

    if not metrics["exportable"]:
        raise RuntimeError(f"raceline failed geometry checks: {metrics}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # store: s(progress)_m, x_m, y_m, v_mps, e(deviation)_m
    np.savetxt(output, values, delimiter=",", header="s_m,x_m,y_m,v_mps,deviation_m", fmt="%.9f")
    load_raceline_csv(output, scale=track_config.scale)
    
    print(f"Saved {output} ({args.opt_type}: {metrics['objective']})")
    print(f"Source-coordinate length {metrics['length_m']:.3f}; maximum curvature excess "
          f"{metrics['curvature_excess_1pm']:.4f} 1/m")
    if metrics["minimum_body_clearance_m"] < 0:
        print(f"WARNING: sampled body-clearance shortfall "
              f"{-metrics['minimum_body_clearance_m']:.9f} in source units")


if __name__ == "__main__":
    main()
