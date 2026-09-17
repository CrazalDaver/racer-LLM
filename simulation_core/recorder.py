"""Trace and plotting-data collection for one race."""

import numpy as np

from config import duration_to_steps, params
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
        self.trace["mpc_horizon"] = int(race.mpc_horizon)
        self.trace["mpc_formulation"] = race.resolved_mpc_formulation

        self.traj_x, self.traj_y, self.velocities = [], [], []
        self.lateral_errors, self.longitudinal_positions = [], []
        self.all_traj_x = [[] for _ in range(race.num_cars)]
        self.all_traj_y = [[] for _ in range(race.num_cars)]
        self.mpc_losses, self.mpc_track_losses = [], []
        self.mpc_action_losses, self.mpc_velocity_losses = [], []
        self.mpc_heading_losses = []
        self.throttle_history, self.steering_history = [], []
        self.frenet_ref_lines = []

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
