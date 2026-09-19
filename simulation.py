import argparse
from pathlib import Path

import numpy as np

from config import duration_to_steps, make_vehicle_params, params
from mpc_controller import MPCController
from mpc_upstream_simple import UpstreamSimpleMPCController
from visualization.plotting import plot_simulation_results
from policy import progress_delta
from simulation_core.environment import RaceEnvironment
from simulation_core.recorder import TraceRecorder
from state import CarState
from simulation_core.config import LLM_OBSERVATION_VARIANTS, SimulationConfig
from simulation_core.utils import (
    create_policy,
    create_vehicle_model,
    per_car_option as _per_car_option,
)
from track import Track, configure_track, load_centerline_csv, load_raceline_csv

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _plot_race_result(trace, track, plot_data):
    """Render the optional interactive summary after recording is complete."""
    plot_simulation_results(plot_data["waypoints"], plot_data["raceline"], plot_data["traj_x"], plot_data["traj_y"], plot_data["velocities"], plot_data["lateral_errors"], plot_data["mpc_losses"], plot_data["mpc_track_losses"], plot_data["mpc_action_losses"], plot_data["mpc_velocity_losses"], plot_data["mpc_heading_losses"], plot_data["longitudinal_positions"], plot_data["frenet_ref_lines"], params, throttle_history=plot_data["throttle_history"], steering_history=plot_data["steering_history"], other_trajectories=[(plot_data["all_traj_x"][i], plot_data["all_traj_y"][i]) for i in range(plot_data["num_cars"]) if i != plot_data["ego_idx"]], ref_speed_history=trace["ref_speed"], ref_lateral_errors=trace["ref_py"], ref_longitudinal_positions=trace["ref_px"], policy_params_history=trace["policy_params"], policy_names=trace["policy_names"], policy_mode_history=trace["policy_mode"], track_length=track.total_length, velocity_track_weight=plot_data["mpc_velocity_track_weight"], heading_track_weight=params.HEADING_TRACK_WEIGHT)


class Race:
    """A configured race that owns its physical environment and main loop."""

    def __init__(self, config: SimulationConfig):
        if not isinstance(config, SimulationConfig):
            raise TypeError("Race expects a SimulationConfig instance")
        self.config = config

        # ==================== 1. SCENARIO AND START GRID ====================
        self._initialize_scenario()

        # ==================== 2. CARS, CONTROLLERS, AND POLICIES ============
        self._initialize_cars()

        # ==================== 3. PHYSICS, RECORDING, AND CONSOLE ============
        self._initialize_runtime()

    def _initialize_scenario(self):
        """Load the track and resolve the roster and initial grid."""
        config = self.config
        if config.track_name is not None:
            configure_track(params, config.track_name, scale=config.track_scale)
        elif config.track_scale is not None:
            params.TRACK_SCALE = float(config.track_scale)

        self.waypoints = load_centerline_csv(
            params.TRACK_CSV, scale=params.TRACK_SCALE
        )
        self.selected_raceline = (
            params.RACELINE_CSV if config.raceline is None else str(config.raceline)
        )
        self.raceline = load_raceline_csv(
            self.selected_raceline, scale=params.TRACK_SCALE
        )
        self.track = Track(self.waypoints)

        self.resolved_policy_names = list(config.validated_policies())
        self.num_cars = len(self.resolved_policy_names)
        self.ego_position = int(config.ego_position)
        if self.ego_position < 1 or self.ego_position > self.num_cars:
            raise ValueError(
                f"ego_position must be between 1 and {self.num_cars}, "
                f"got {self.ego_position}"
            )
        self.ego_idx = 0
        ego_start_rank = self.ego_position - 1
        remaining_ranks = [
            rank for rank in range(self.num_cars) if rank != ego_start_rank
        ]
        self.start_ranks = [ego_start_rank, *remaining_ranks]
        self.start_gap = (
            params.CAR_START_GAP
            if config.start_gap is None
            else float(config.start_gap)
        )
        if self.start_gap <= 0.0:
            raise ValueError(f"start_gap must be positive, got {self.start_gap}")

        def initial_values(values, default, name):
            if values is None:
                return np.full(self.num_cars, default, dtype=float)
            array = np.asarray(values, dtype=float)
            if array.shape != (self.num_cars,):
                raise ValueError(
                    f"{name} must contain {self.num_cars} values, "
                    f"got shape {array.shape}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must contain only finite values")
            return array

        self.initial_longitudinal_offsets = initial_values(
            config.start_s_offsets, 0.0, "start_s_offsets"
        )
        self.initial_lateral_offsets = initial_values(
            config.start_e_offsets, 0.0, "start_e_offsets"
        )
        self.initial_speeds = initial_values(
            config.start_speeds, float(config.initial_speed), "start_speeds"
        )
        if np.any(self.initial_speeds < 0.0):
            raise ValueError("start_speeds must be non-negative")

        for name, values in (
            ("alpha_weight_paths", config.alpha_weight_paths),
            ("llm_models", config.llm_models),
            ("llm_temperatures", config.llm_temperatures),
        ):
            if values is not None and len(values) != self.num_cars:
                raise ValueError(
                    f"{name} must contain {self.num_cars} values, got {len(values)}"
                )
        self.resolved_llm_model_names = [
            _per_car_option(config.llm_models, idx, config.llm_model)
            for idx in range(self.num_cars)
        ]
        self.resolved_llm_temperatures = [
            _per_car_option(config.llm_temperatures, idx, config.llm_temperature)
            for idx in range(self.num_cars)
        ]

    def _initialize_cars(self):
        """Construct each car's controller, policy, plant, and initial state."""
        config = self.config
        self.vehicle_model_type = (
            params.VEHICLE_MODEL
            if config.vehicle_model is None
            else config.vehicle_model
        ).lower()
        vehicle_params = make_vehicle_params(self.vehicle_model_type)
        self.mpc_horizon = (
            params.MPC_HORIZON if config.mpc_horizon is None else int(config.mpc_horizon)
        )
        self.mpc_velocity_track_weight = (
            params.VELOCITY_TRACK_WEIGHT
            if config.mpc_velocity_weight is None
            else float(config.mpc_velocity_weight)
        )
        if self.mpc_velocity_track_weight < 0.0:
            raise ValueError("velocity_track_weight must be non-negative")
        self.mpc_track_slack_weight = (
            params.MPC_TRACK_SLACK_WEIGHT
            if config.mpc_track_slack is None
            else float(config.mpc_track_slack)
        )
        if self.mpc_track_slack_weight < 0.0:
            raise ValueError("track_slack_weight must be non-negative")
        self.mpc_collision_slack_weight = (
            params.MPC_COLLISION_SLACK_WEIGHT
            if config.mpc_collision_slack is None
            else float(config.mpc_collision_slack)
        )
        if self.mpc_collision_slack_weight < 0.0:
            raise ValueError("collision_slack_weight must be non-negative")
        self.resolved_mpc_formulation = (
            params.MPC_FORMULATION
            if config.mpc_formulation is None
            else str(config.mpc_formulation)
        )
        controller_types = {
            "constrained_local": MPCController,
            "upstream_simple": UpstreamSimpleMPCController,
        }
        if self.resolved_mpc_formulation not in controller_types:
            raise ValueError(
                "mpc_formulation must be one of "
                f"{sorted(controller_types)}, got {self.resolved_mpc_formulation!r}"
            )
        controller_type = controller_types[self.resolved_mpc_formulation]
        if self.resolved_mpc_formulation == "upstream_simple":
            self.mpc_velocity_track_weight = 0.0

        self.mpcs, self.policies = [], []
        self.vehicle_models, self.states = [], []
        for car_idx, policy_name in enumerate(self.resolved_policy_names):
            car_mpc_horizon = (
                params.IBR_HORIZON
                if config.mpc_horizon is None and policy_name == "ibr"
                else self.mpc_horizon
            )
            mpc = controller_type(
                dt=params.MPC_DT,
                horizon=car_mpc_horizon,
                Q=params.Q,
                R=params.R,
                max_vel_rate=params.MAX_VEL_RATE,
                max_steer_delta=params.MPC_MAX_STEER_RATE * params.MPC_DT,
                db_params=vehicle_params,
                track_width=params.TRACK_WIDTH,
                track_constraint_half_width=params.MPC_CONSTRAINT_HALF_WIDTH,
                max_state_speed=params.MAX_STATE_SPEED,
                velocity_track_weight=self.mpc_velocity_track_weight,
                heading_track_weight=params.HEADING_TRACK_WEIGHT,
                track_length=self.track.total_length,
                min_longitudinal_separation=params.MPC_MIN_LONGITUDINAL_SEPARATION,
                min_lateral_separation=params.MPC_MIN_LATERAL_SEPARATION,
                max_constrained_opponents=params.MPC_MAX_CONSTRAINED_OPPONENTS,
                collision_superellipse_exponent=(
                    params.MPC_COLLISION_SUPERELLIPSE_EXPONENT
                ),
                enforce_track_boundary=params.MPC_ENFORCE_TRACK_BOUNDARY,
                enforce_collision_avoidance=params.MPC_ENFORCE_COLLISION_AVOIDANCE,
                track_slack_weight=self.mpc_track_slack_weight,
                collision_slack_weight=self.mpc_collision_slack_weight,
                enforce_friction_speed_limit=(
                    params.MPC_ENFORCE_FRICTION_SPEED_LIMIT
                ),
                friction_coefficient=params.MPC_FRICTION_COEFFICIENT,
                track=self.track,
            )
            self.mpcs.append(mpc)
            self.policies.append(
                create_policy(
                    policy_name,
                    self.track,
                    mpc,
                    raceline=self.raceline,
                    car_idx=car_idx,
                    static_policy_params=(
                        config.static_params
                        if config.static_params_by_car is None
                        else np.asarray(config.static_params_by_car, dtype=float)[car_idx]
                    ),
                    alpharacer_params=config.alpha_params,
                    alpharacer_weight_path=(
                        config.alpha_weight_path
                        if config.alpha_weight_paths is None
                        else _per_car_option(
                            config.alpha_weight_paths,
                            car_idx,
                            config.alpha_weight_path,
                        )
                    ),
                    ibr_implementation=config.ibr_implementation,
                    llm_policy_params=config.llm_params,
                    llm_output_mode=config.llm_output_mode,
                    llm_model_name=self.resolved_llm_model_names[car_idx],
                    llm_temperature=self.resolved_llm_temperatures[car_idx],
                    llm_update_interval_seconds=config.llm_update_interval,
                    llm_observation_variant=config.llm_observation_variant,
                    self_play_rl_model_path=config.rl_model_path,
                    self_play_rl_backend=config.rl_backend,
                    self_play_rl_strict_load=config.rl_strict_load,
                    self_play_rl_action_order=config.rl_action_order,
                )
            )
            self.vehicle_models.append(
                create_vehicle_model(
                    self.vehicle_model_type,
                    dt=params.SIM_DT,
                    db_params=vehicle_params,
                    max_speed=params.MAX_STATE_SPEED,
                )
            )

            start_s = (
                -self.start_ranks[car_idx] * self.start_gap
                + self.initial_longitudinal_offsets[car_idx]
            )
            start_e = self.initial_lateral_offsets[car_idx]
            start_x, start_y = self.track.frenet_to_world(start_s, start_e)
            _, _, start_psi = self.track.interpolate(start_s)
            self.states.append(
                CarState(
                    x=start_x,
                    y=start_y,
                    vx=self.initial_speeds[car_idx],
                    vy=0.0,
                    psi=start_psi,
                    omega=0.0,
                )
            )

        self.ibr_implementation = config.ibr_implementation
        self.llm_update_interval_seconds = config.llm_update_interval
        self.total_time = config.duration
        self.save_trace = config.save_trace
        self.show_plot = config.show_plot
        self.quiet = config.quiet

    def _initialize_runtime(self):
        """Create the environment and the two output observers."""
        simulation_time = (
            params.TOTAL_TIME if self.total_time is None else float(self.total_time)
        )
        self.total_steps = duration_to_steps(simulation_time, params.SIM_DT)
        report_steps = duration_to_steps(
            params.PROGRESS_REPORT_INTERVAL_SECONDS, params.SIM_DT
        )
        initial_progress = [
            -self.start_ranks[idx] * self.start_gap
            + self.initial_longitudinal_offsets[idx]
            for idx in range(self.num_cars)
        ]
        self.environment = RaceEnvironment(
            track=self.track,
            states=self.states,
            vehicle_models=self.vehicle_models,
            initial_progress=initial_progress,
        )
        self.recorder = TraceRecorder(self)
        self.reporter = _ProgressReporter(
            self, self.total_steps, report_steps
        ).start()

    def run(self):
        """Run the visible observe, act, step, and record race loop."""

        # ==================== 1. ADVANCE THE RACE ======================
        for step in range(self.total_steps):
            # ----- A. Capture the current scene -----
            self.environment.begin_step(step)
            observation = self.environment.capture(self.policies)

            # ----- B. Ask every policy to act -----
            actions = tuple(
                policy.get_action(observation) for policy in self.policies
            )

            # ----- C. Advance physics by one step -----
            transition = self.environment.step(actions)

            # ----- D. Record and report the result -----
            self.recorder.record(step, transition, self.environment)
            self.reporter.report_step(step, self.environment, transition)

        # ==================== 2. FINALIZE OUTPUTS ======================
        self.reporter.finish()
        trace = self.recorder.postprocess()
        if self.show_plot:
            _plot_race_result(trace, self.track, self.recorder.plot_data())
        return trace


class _ProgressReporter:
    """Keep console/progress-bar concerns outside the simulation loop."""

    def __init__(self, race, total_steps, report_interval):
        self.race = race
        self.total_steps = total_steps
        self.report_interval = report_interval
        self.bar = None

    def start(self):
        race = self.race
        if not race.quiet:
            print(
                "Simulation setup: "
                f"track={params.TRACK_CSV}, scale={params.TRACK_SCALE:g}, "
                f"half_width={params.TRACK_HALF_WIDTH:g}, "
                f"policies={race.resolved_policy_names}, "
                f"ego_start_position={race.ego_position}/{race.num_cars}, "
                f"vehicle_model={race.vehicle_model_type}, "
                f"mpc_formulation={race.resolved_mpc_formulation}, "
                f"step={params.SIM_DT:g}s"
            )
        if not race.quiet and any(
            name == "llm_alpharacer" for name in race.resolved_policy_names
        ):
            print(f"LLM models by car: {race.resolved_llm_model_names}")
            print(f"LLM temperatures by car: {race.resolved_llm_temperatures}")
        if tqdm is not None and not race.quiet:
            self.bar = tqdm(
                total=self.total_steps,
                desc="Simulation",
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )
        log_fn = self.bar.write if self.bar is not None else print
        for policy in race.policies:
            policy.log_fn = log_fn
        return self

    def _write(self, message):
        if self.bar is not None:
            self.bar.write(message)
        else:
            print(message)

    def report_step(self, step, environment, transition):
        if self.bar is not None:
            self.bar.update(1)
        if self.race.quiet or (step + 1) % self.report_interval != 0:
            return
        ego_idx = self.race.ego_idx
        position = 1 + sum(
            environment.progress_s[idx] > environment.progress_s[ego_idx]
            for idx in range(self.race.num_cars)
            if idx != ego_idx
        )
        action = transition.actions[ego_idx]
        self._write(
            f"Step {step+1}/{self.total_steps}, "
            f"t={(step+1)*params.SIM_DT:.1f}s, "
            f"position={position}/{self.race.num_cars}, "
            f"vx={environment.states[ego_idx].vx:.2f} m/s, "
            f"e={transition.next_e[ego_idx]:.3f}, "
            f"throttle={action.throttle:.2f}, steer={action.steer:.2f}"
        )

    def finish(self):
        if self.bar is not None:
            self.bar.close()


def run_simulation(config: SimulationConfig):
    """Construct and run one race from its explicit configuration."""
    return Race(config).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the default racer simulation scenario.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["llm_alpharacer", "ibr", "ibr"],
        help=(
            "One policy per car, e.g. --policies ibr ibr ibr or "
            "--policies parameter_ibr parameter_ibr parameter_ibr."
        ),
    )
    parser.add_argument(
        "--ibr-implementation",
        choices=["local_wang", "historical_strict"],
        default="local_wang",
        help="IBR solver implementation (default: local_wang).",
    )
    parser.add_argument("--parameter-ibr-rounds", type=int, default=None, metavar="N")
    parser.add_argument("--parameter-ibr-search-points", type=int, default=None, metavar="N")
    parser.add_argument(
        "--parameter-ibr-update-interval", type=float, default=None, metavar="SECONDS"
    )
    parser.add_argument("--ego-position", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument(
        "--llm-observation-variant",
        choices=LLM_OBSERVATION_VARIANTS,
        default="baseline",
    )
    parser.add_argument("--total-time", type=float, default=None, metavar="SECONDS")
    parser.add_argument(
        "--mpc-velocity-track-weight",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Override the shared MPC reference-speed tracking weight.",
    )
    parser.add_argument(
        "--mpc-track-slack-weight",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Override the quadratic MPC track-boundary slack weight.",
    )
    parser.add_argument(
        "--mpc-collision-slack-weight",
        type=float,
        default=None,
        metavar="WEIGHT",
        help="Override the quadratic MPC collision-avoidance slack weight.",
    )
    parser.add_argument(
        "--mpc-formulation",
        choices=["constrained_local", "upstream_simple"],
        default=None,
        help="Select the local constrained or detachable upstream-style MPC.",
    )
    parser.add_argument(
        "--raceline-csv",
        default=None,
        metavar="CSV",
        help="Override the registered track raceline without changing config.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="NPZ",
        help="Trace output path (default: config TRACE_OUTPUT).",
    )
    parser.add_argument("--start-gap", type=float, default=None, metavar="METERS")
    parser.add_argument(
        "--initial-speeds",
        nargs="+",
        type=float,
        default=None,
        metavar="MPS",
        help="Initial speed for each car, in the same order as --policies.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Do not open the result plot.")
    cli_args = parser.parse_args()

    output_path = None
    if cli_args.output is not None:
        output_path = Path(cli_args.output).expanduser()
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        cli_args.mpc_track_slack_weight is not None
        and cli_args.mpc_track_slack_weight < 0.0
    ):
        parser.error("--mpc-track-slack-weight must be non-negative")
    if (
        cli_args.mpc_collision_slack_weight is not None
        and cli_args.mpc_collision_slack_weight < 0.0
    ):
        parser.error("--mpc-collision-slack-weight must be non-negative")

    if cli_args.parameter_ibr_rounds is not None:
        if cli_args.parameter_ibr_rounds < 1:
            parser.error("--parameter-ibr-rounds must be at least 1")
        params.PARAMETER_IBR_ROUNDS = cli_args.parameter_ibr_rounds
    if cli_args.parameter_ibr_search_points is not None:
        if cli_args.parameter_ibr_search_points < 3:
            parser.error("--parameter-ibr-search-points must be at least 3")
        params.PARAMETER_IBR_SEARCH_POINTS = cli_args.parameter_ibr_search_points
    if cli_args.parameter_ibr_update_interval is not None:
        if cli_args.parameter_ibr_update_interval < 0.0:
            parser.error("--parameter-ibr-update-interval must be non-negative")
        params.PARAMETER_IBR_UPDATE_INTERVAL_SECONDS = cli_args.parameter_ibr_update_interval

    # Use canonical policy names only, e.g. "alpharacer", "static_alpharacer",
    # "ibr", "parameter_ibr", or "llm_alpharacer".  Aliases are intentionally
    # rejected so saved artifacts always identify one unambiguous method.
    # Tracks: "berlin" (scale 2), "modena" (scale 2), or "rounded_rectangle" (scale 1).

    # Wang-style trajectory IBR:
    # .venv/bin/python simulation.py --policies ibr ibr ibr --no-plot
    # Five-parameter IBR using the same low-level MPC as alpha-RACER/LLM:
    # .venv/bin/python simulation.py --policies parameter_ibr parameter_ibr parameter_ibr --no-plot

    config = SimulationConfig(
        policies=cli_args.policies,
        duration=cli_args.total_time,
        ego_position=cli_args.ego_position,
        raceline=cli_args.raceline_csv,
        start_gap=cli_args.start_gap,
        start_speeds=cli_args.initial_speeds,
        initial_speed=0.0,
        mpc_formulation=cli_args.mpc_formulation,
        mpc_velocity_weight=cli_args.mpc_velocity_track_weight,
        mpc_track_slack=cli_args.mpc_track_slack_weight,
        mpc_collision_slack=cli_args.mpc_collision_slack_weight,
        ibr_implementation=cli_args.ibr_implementation,
        llm_observation_variant=cli_args.llm_observation_variant,
        trace_output=output_path,
        show_plot=not cli_args.no_plot,
    )
    run_simulation(config)
