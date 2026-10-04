"""Trace and plotting-data collection for one race."""

from pathlib import Path
import subprocess

import numpy as np

from config import duration_to_steps, params
from track import TRACKS
from .trace import (
    create_simulation_trace,
    record_simulation_step,
    save_simulation_trace,
)


class TraceRecorder:
    """Observe transitions without participating in policy or physics logic."""

    def __init__(self, race):
        self.race = race
        self.ego_idx = race.ego_idx
        self.num_cars = race.num_cars
        self.track_half_width = getattr(
            params, "TRACK_HALF_WIDTH", params.TRACK_WIDTH / 2.0
        )
        self.offroad_recovery_threshold = params.OFFROAD_RECOVERY_THRESHOLD
        self.ref_debug_sample_steps = duration_to_steps(
            params.REF_DEBUG_SAMPLE_INTERVAL_SECONDS, params.SIM_DT
        )
        self.trajectory_sample_steps = duration_to_steps(
            params.TRAJECTORY_SAMPLE_INTERVAL_SECONDS, params.SIM_DT
        )
        self.ref_component_horizon = max(mpc.horizon for mpc in race.mpcs) + 1

        self.trace = create_simulation_trace(
            resolved_policy_names=race.resolved_policy_names,
            policies=race.policies,
            ego_idx=race.ego_idx,
            ibr_implementation=race.ibr_implementation,
            resolved_llm_model_names=race.resolved_llm_model_names,
            resolved_llm_temperatures=race.resolved_llm_temperatures,
            num_cars=race.num_cars,
            llm_update_interval_seconds=race.llm_update_interval_seconds,
            ego_position=race.ego_position,
            start_ranks=race.start_ranks,
            start_gap=race.start_gap,
            initial_longitudinal_offsets=race.initial_longitudinal_offsets,
            initial_lateral_offsets=race.initial_lateral_offsets,
            initial_speeds=race.initial_speeds,
            vehicle_model_type=race.vehicle_model_type,
            mpc_velocity_track_weight=race.mpc_velocity_track_weight,
            mpc_track_slack_weight=race.mpc_track_slack_weight,
            mpc_collision_slack_weight=race.mpc_collision_slack_weight,
        )
        self.trace["raceline_csv"] = race.selected_raceline
        core_deviations = self._protocol_core_deviations(race)
        scenario_deviations = self._protocol_scenario_deviations(race)
        self.trace["protocol_reference_version"] = params.PROTOCOL_VERSION
        self.trace["protocol_version"] = (
            params.PROTOCOL_VERSION if not core_deviations else "unversioned"
        )
        self.trace["protocol_core_deviations"] = core_deviations
        self.trace["protocol_scenario_deviations"] = scenario_deviations
        self.trace["protocol_core_compatible"] = not core_deviations
        self.trace["protocol_evaluation_scenario_compatible"] = (
            not core_deviations and not scenario_deviations
        )
        git_provenance = self._git_provenance()
        self.trace.update(git_provenance)
        self.trace["mpc_horizon"] = int(race.mpc_horizon)
        self.trace["mpc_formulation"] = race.resolved_mpc_formulation
        self.trace["random_seed"] = race.random_seed
        self.trace["oracle_settings"] = (
            {}
            if race.oracle_evaluator is None
            else race.oracle_evaluator.settings_dict()
        )
        self.trace["oracle_mode_names"] = list(params.LLM_MODE_PRESETS)
        self.trace["oracle_mode_presets"] = {
            name: np.asarray(values, dtype=float).tolist()
            for name, values in params.LLM_MODE_PRESETS.items()
        }

        self.traj_x, self.traj_y, self.velocities = [], [], []
        self.lateral_errors, self.longitudinal_positions = [], []
        self.all_traj_x = [[] for _ in range(race.num_cars)]
        self.all_traj_y = [[] for _ in range(race.num_cars)]
        self.mpc_losses, self.mpc_track_losses = [], []
        self.mpc_action_losses, self.mpc_velocity_losses = [], []
        self.mpc_heading_losses = []
        self.throttle_history, self.steering_history = [], []
        self.frenet_ref_lines = []

    @staticmethod
    def _git_provenance():
        """Capture the code identity needed to classify formal versus exploratory runs."""
        repository = Path(__file__).resolve().parents[1]

        def git(*args):
            result = subprocess.run(
                ["git", *args],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

        try:
            commit = git("rev-parse", "HEAD")
            branch = git("rev-parse", "--abbrev-ref", "HEAD")
            status = git("status", "--porcelain", "--untracked-files=all")
            error = ""
        except (OSError, subprocess.CalledProcessError) as exc:
            commit = "unknown"
            branch = "unknown"
            status = "unknown"
            error = str(exc)
        return {
            "git_commit_sha": commit,
            "git_branch": branch,
            "git_dirty": status != "",
            "git_status_porcelain": status,
            "git_provenance_error": error,
        }

    @staticmethod
    def _protocol_core_deviations(race):
        """List runtime choices that may not carry the frozen protocol label."""
        deviations = []

        def check(name, actual, expected):
            if isinstance(expected, float):
                matches = bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
            else:
                matches = actual == expected
            if not matches:
                deviations.append(f"{name}={actual!r} (expected {expected!r})")

        check("vehicle_model", race.vehicle_model_type, "kinematic")
        check("mpc_formulation", race.resolved_mpc_formulation, "constrained_local")
        check("mpc_horizon", race.mpc_horizon, params.MPC_HORIZON)
        check("velocity_track_weight", race.mpc_velocity_track_weight, 10.0)
        check("track_slack_weight", race.mpc_track_slack_weight, 1.0e3)
        check("collision_slack_weight", race.mpc_collision_slack_weight, 1.0e4)
        effective_llm_interval = (
            params.LLM_UPDATE_INTERVAL_SECONDS
            if race.llm_update_interval_seconds is None
            else race.llm_update_interval_seconds
        )
        check("llm_update_interval_seconds", effective_llm_interval, 1.0)

        selected_pair = (params.TRACK_CSV, race.selected_raceline)
        selected_spec = next(
            (
                spec
                for spec in TRACKS.values()
                if (spec.centerline, spec.raceline) == selected_pair
            ),
            None,
        )
        if selected_spec is None:
            deviations.append(
                "track/raceline pair is not registered in K-Constrained-v1: "
                f"{selected_pair!r}"
            )
        else:
            check("track_scale", params.TRACK_SCALE, float(selected_spec.scale))
            check(
                "track_half_width",
                params.TRACK_HALF_WIDTH,
                float(selected_spec.half_width),
            )
            check(
                "mpc_constraint_half_width",
                params.MPC_CONSTRAINT_HALF_WIDTH,
                float(selected_spec.half_width),
            )
        return deviations

    @staticmethod
    def _protocol_scenario_deviations(race):
        """List departures from one cell of the canonical 3x3 evaluation grid."""
        deviations = []
        simulation_time = (
            params.TOTAL_TIME if race.total_time is None else float(race.total_time)
        )
        expected = (
            ("num_cars", race.num_cars, 3),
            ("duration_s", simulation_time, 50.0),
            ("start_gap_m", race.start_gap, 6.0),
        )
        for name, actual, target in expected:
            if not np.isclose(float(actual), float(target), rtol=0.0, atol=1e-12):
                deviations.append(f"{name}={actual!r} (expected {target!r})")
        if race.random_seed != 0:
            deviations.append(f"random_seed={race.random_seed!r} (expected 0)")
        for name, actual in (
            ("initial_longitudinal_offsets", race.initial_longitudinal_offsets),
            ("initial_lateral_offsets", race.initial_lateral_offsets),
            ("initial_speeds", race.initial_speeds),
        ):
            if np.shape(actual) != (3,) or not np.allclose(actual, 0.0):
                deviations.append(f"{name} must be three zeros")
        if race.ego_position not in (1, 2, 3):
            deviations.append("ego_position must be one of 1, 2, 3")
        return deviations

    def _ref_component_snapshot(self, policy, name):
        values = np.full(self.ref_component_horizon, np.nan, dtype=float)
        components = getattr(policy, "last_ref_components", None)
        if components is None or name not in components:
            return values
        source = np.asarray(components[name], dtype=float).reshape(-1)
        count = min(source.size, self.ref_component_horizon)
        values[:count] = source[:count]
        return values

    def record(self, step, transition, environment):
        race = self.race
        ref_fx, ref_fy = record_simulation_step(
            self.trace,
            step=step,
            track=race.track,
            policies=race.policies,
            mpcs=race.mpcs,
            states=transition.trace_states,
            actions=transition.actions,
            car_s=transition.previous_s,
            car_e=transition.previous_e,
            ego_idx=self.ego_idx,
            track_half_width=self.track_half_width,
            offroad_recovery_threshold=self.offroad_recovery_threshold,
            control_update=True,
            offroad_recovery_active=transition.offroad_active,
            offroad_recovery_heading_error=transition.offroad_heading_errors,
            ref_component_snapshot=self._ref_component_snapshot,
        )
        self.trace["collision"][-1] = transition.contact_happened
        self.trace["collision_overlap"][-1] = transition.contact_max_overlap

        ego_policy = race.policies[self.ego_idx]
        ego_mpc = race.mpcs[self.ego_idx]
        ego_trace_state = transition.trace_states[self.ego_idx]
        if (
            (step + 1) % self.ref_debug_sample_steps == 0
            and ego_policy.last_ref_traj is not None
        ):
            self.frenet_ref_lines.append(
                (ego_trace_state.x, ego_trace_state.y, ref_fx, ref_fy)
            )

        self.mpc_losses.append(ego_mpc.last_loss)
        self.mpc_track_losses.append(ego_mpc.last_track_loss)
        self.mpc_action_losses.append(ego_mpc.last_action_loss)
        self.mpc_velocity_losses.append(ego_mpc.last_velocity_loss)
        self.mpc_heading_losses.append(ego_mpc.last_heading_loss)
        self.throttle_history.append(transition.actions[self.ego_idx].throttle)
        self.steering_history.append(transition.actions[self.ego_idx].steer)
        self.lateral_errors.append(transition.next_e[self.ego_idx])
        self.longitudinal_positions.append(environment.progress_s[self.ego_idx])

        if step % self.trajectory_sample_steps == 0:
            for idx, state in enumerate(environment.states):
                self.all_traj_x[idx].append(state.x)
                self.all_traj_y[idx].append(state.y)
            ego_state = environment.states[self.ego_idx]
            self.traj_x.append(ego_state.x)
            self.traj_y.append(ego_state.y)
            self.velocities.append(ego_state.vx)

    def postprocess(self):
        """Complete metadata, optionally save the NPZ, and return the trace."""
        final_time = float(self.trace["time"][-1]) if self.trace["time"] else 0.0
        for policy in self.race.policies:
            policy.flush_solver_warnings(sim_time=final_time)

        histories = [
            getattr(policy, "llm_raw_io_history", [])
            for policy in self.race.policies
        ]
        max_calls = max((len(history) for history in histories), default=0)
        fields = {
            "llm_raw_call_indices": (lambda item: item["call"], -1),
            "llm_raw_prompts": (lambda item: item["prompt"], ""),
            "llm_raw_responses": (lambda item: item["response"], ""),
            "llm_raw_errors": (lambda item: item["error"], ""),
            "llm_raw_done_reasons": (
                lambda item: item.get("done_reason", ""), ""
            ),
            "llm_raw_eval_counts": (lambda item: item.get("eval_count", -1), -1),
            "llm_raw_prompt_eval_counts": (
                lambda item: item.get("prompt_eval_count", -1), -1
            ),
            "llm_raw_latency_seconds": (
                lambda item: item.get("latency_seconds", np.nan), np.nan
            ),
        }
        for key, (value, padding) in fields.items():
            self.trace[key] = [
                [padding] * (max_calls - len(history))
                + [value(item) for item in history]
                for history in histories
            ]
        self.trace["car_policy_decision_counts"] = [
            int(getattr(policy, "policy_decision_count", 0))
            for policy in self.race.policies
        ]
        oracle_histories = [
            list(getattr(policy, "oracle_decision_history", []))
            for policy in self.race.policies
        ]
        self.trace["oracle_decisions_by_car"] = oracle_histories
        self.trace["oracle_mode_counts_by_car"] = [
            {
                mode: sum(
                    decision.get("selected_mode") == mode
                    for decision in history
                )
                for mode in params.LLM_MODE_PRESETS
            }
            for history in oracle_histories
        ]

        terminal_s, terminal_e = self.race.environment.project_current()
        self.trace["terminal_car_progress"] = [float(value) for value in terminal_s]
        self.trace["terminal_car_lateral_offset"] = [
            float(value) for value in terminal_e
        ]
        self.trace["initial_car_progress"] = [
            -self.race.start_ranks[idx] * self.race.start_gap
            + self.race.initial_longitudinal_offsets[idx]
            for idx in range(self.race.num_cars)
        ]

        if self.race.save_trace:
            save_simulation_trace(
                self.trace,
                self.race.track,
                params,
                output_path=self.race.config.trace_output,
            )
        elif not self.race.quiet:
            print("Skipped saving simulation trace")
        return self.trace

    def plot_data(self):
        race = self.race
        return {
            "waypoints": race.waypoints,
            "raceline": race.raceline,
            "traj_x": self.traj_x,
            "traj_y": self.traj_y,
            "velocities": self.velocities,
            "lateral_errors": self.lateral_errors,
            "mpc_losses": self.mpc_losses,
            "mpc_track_losses": self.mpc_track_losses,
            "mpc_action_losses": self.mpc_action_losses,
            "mpc_velocity_losses": self.mpc_velocity_losses,
            "mpc_heading_losses": self.mpc_heading_losses,
            "longitudinal_positions": self.longitudinal_positions,
            "frenet_ref_lines": self.frenet_ref_lines,
            "throttle_history": self.throttle_history,
            "steering_history": self.steering_history,
            "all_traj_x": self.all_traj_x,
            "all_traj_y": self.all_traj_y,
            "num_cars": race.num_cars,
            "ego_idx": race.ego_idx,
            "mpc_velocity_track_weight": race.mpc_velocity_track_weight,
        }
