"""Trace and plotting-data collection for one race."""

from pathlib import Path
import subprocess
import hashlib
import json
from dataclasses import asdict

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
            mpc_min_state_speed=race.mpc_min_state_speed,
            mpc_velocity_track_weight=race.mpc_velocity_track_weight,
            mpc_track_slack_weight=race.mpc_track_slack_weight,
            mpc_collision_slack_weight=race.mpc_collision_slack_weight,
        )
        self.trace["raceline_csv"] = race.selected_raceline
        self.trace["development_status"] = params.DEVELOPMENT_STATUS
        git_provenance = self._git_provenance()
        self.trace.update(git_provenance)
        root=Path(__file__).resolve().parents[1]
        self.trace['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (root,root/'simulation_core',root/'policy',root/'policy/alpharacer',root/'track') for p in folder.glob('*.py')}
        self.trace['input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in map(Path,(params.TRACK_CSV,race.selected_raceline))}
        self.trace['vehicle_parameters']=vars(race.vehicle_models[0].p).copy()
        self.trace['initial_states']=[state.to_array().tolist() for state in race.environment.states]
        self.trace['simulation_config']=asdict(race.config)
        self.trace['result_class']='exploratory'
        self.trace['mpc_prediction_model']='kinematic'
        self.trace['mpc_prediction_states']=4
        self.trace['mpc_settings']={name:(value.tolist() if isinstance(value,np.ndarray) else value)
            for name in ('dt','horizon','Q','R','min_speed','max_speed','velocity_track_weight',
                'heading_track_weight','enforce_track_boundary','enforce_collision_avoidance',
                'collision_superellipse_exponent','friction_coefficient','enforce_friction_speed_limit')
            if (value:=getattr(race.mpcs[0],name,None)) is not None}
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
        self.trace["potential_regret_settings"] = (
            {}
            if race.potential_regret_evaluator is None
            else race.potential_regret_evaluator.settings_dict()
        )

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
        self.trace["car_collision"].append(list(transition.contact_active))

        self.trace.setdefault('car_next_state',[]).append([state.to_array().tolist() for state in environment.states])
        self.trace.setdefault('car_next_px',[]).append(list(transition.next_s))
        self.trace.setdefault('car_next_py',[]).append(list(transition.next_e))
        self.trace.setdefault('car_requested_action',[]).append([getattr(policy,'last_requested_action',np.array([action.throttle,action.steer])).tolist() for policy,action in zip(race.policies,transition.actions)])
        self.trace.setdefault('car_mpc_solve_seconds',[]).append([m.last_diagnostics['solve_seconds'] for m in race.mpcs])
        self.trace.setdefault('car_mpc_constraint_residual',[]).append([m.last_diagnostics.get('residual') if m.last_diagnostics.get('residual') is not None else np.nan for m in race.mpcs])
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
            "llm_raw_prompt_eval_cached_counts": (
                lambda item: item.get("prompt_eval_cached_count", -1), -1
            ),
            "llm_raw_prompt_eval_durations_ns": (
                lambda item: item.get("prompt_eval_duration_ns", -1), -1
            ),
            "llm_raw_load_durations_ns": (
                lambda item: item.get("load_duration_ns", -1), -1
            ),
            "llm_raw_eval_durations_ns": (
                lambda item: item.get("eval_duration_ns", -1), -1
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
        mode_alpharacer_histories = [
            list(getattr(policy, "mode_alpharacer_decision_history", []))
            for policy in self.race.policies
        ]
        self.trace["mode_alpharacer_decisions_by_car"] = (
            mode_alpharacer_histories
        )
        self.trace["mode_alpharacer_mode_counts_by_car"] = [
            {
                mode: sum(
                    decision.get("selected_mode") == mode
                    for decision in history
                )
                for mode in params.LLM_MODE_PRESETS
            }
            for history in mode_alpharacer_histories
        ]
        regret_evaluator = self.race.potential_regret_evaluator
        self.trace["potential_regret_decisions"] = (
            [] if regret_evaluator is None else regret_evaluator.decisions
        )
        self.trace["potential_regret_snapshot_validation"] = (
            {}
            if regret_evaluator is None
            else regret_evaluator.snapshot_validation
        )

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

        self.trace['simulated_duration_seconds']=len(self.trace['time'])*params.SIM_DT
        self.trace['terminal_states']=[state.to_array().tolist() for state in self.race.environment.states]
        self.trace['terminal_progress']=list(terminal_s)
        if self.race.save_trace:
            output=Path(self.race.config.trace_output or params.TRACE_OUTPUT)
            output.parent.mkdir(parents=True,exist_ok=True)
            keys=('source_sha256','input_sha256','vehicle_parameters','mpc_settings','initial_states',
                  'simulation_config','git_commit_sha','git_dirty','result_class','development_status',
                  'mpc_prediction_model','mpc_prediction_states')
            metadata={key:self.trace[key] for key in keys}
            metadata['failure']=self.trace.get('failure')
            (output.parent/'manifest.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
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
