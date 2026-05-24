import abc
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from langchain_community.chat_models import ChatOllama

# import sys
# sys.path.append(r'F:\desktop\alpha-RACER')
# from car_dynamics.car_dynamics.models_jax.dbm import DynamicBicycleModel, DynamicParams, CarState, CarAction

# ============================= parameters =============================
"""
alpha-RACER style simulation (with out-of-track penalty)
=========================================================
- Bicycle Kinematic Model
- Track projection and out-of-track deceleration penalty (|e|>0.55 → vx *= exp(-3*(|e|-0.55)))
- Support different control policies: DefaultControlPolicy (sinusoidal steering) and optional LLMControlPolicy
"""

# ============================= parameters =============================
class Params:
    DT = 0.02              # control period (s)
    TOTAL_TIME = 20.0      # total simulation time (s)
    CAR_LEN = 2.0          # wheelbase (m)
    TRACK_WIDTH = 2.0      # track width (m) -> half-width = 1.0 m
    OFFROAD_TOLERANCE = 0.55   # lateral error threshold for penalty (beyond half-width)
    OFFROAD_STRENGTH = 3.0     # penalty strength (exponential decay)
    TARGET_SPEED = 3.5         # max speed at full throttle (m/s)
    MAX_STEER_ANGLE = 0.6      # max steering angle (rad)
    STEER_AMP = 0.4            # amplitude for default policy
    STEER_PERIOD = 6.0         # period for default policy
params = Params()

# ============================= CarState and CarAction =============================


class CarState:
    """Vehicle state with attributes required by control policies"""
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, psi=0.0, omega=0.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.psi = psi      # heading angle (rad)
        self.omega = omega  # yaw rate (rad/s)

    def to_array(self):
        return np.array([self.x, self.y, self.vx, self.vy, self.psi, self.omega])

    @classmethod
    def from_array(cls, arr):
        return cls(x=arr[0], y=arr[1], vx=arr[2], vy=arr[3], psi=arr[4], omega=arr[5])


class CarAction:
    """Control command: target_vel in [-1, 1], target_steer in [-1, 1]"""
    def __init__(self, target_vel=0.0, target_steer=0.0):
        self.target_vel = target_vel
        self.target_steer = target_steer

# ============================= Bicycle Kinematic Model =============================


class BicycleKinematicModel:
    """State: [x, y, vx, vy, psi, psi_dot] but vy always 0 (no sideslip)"""
    def __init__(self, dt=params.DT, L=params.CAR_LEN):
        self.dt = dt
        self.L = L

    def step(self, state: np.ndarray, throttle: float, steer: float) -> np.ndarray:
        x, y, vx, vy, psi, psi_dot = state

        # throttle 0~1 -> target speed
        target_v = throttle * params.TARGET_SPEED
        ax = (target_v - vx) / self.dt * 0.5
        ax = np.clip(ax, -3.0, 3.0)

        vx_new = vx + ax * self.dt
        vx_new = max(0.0, vx_new)

        # steering angle limit
        steer = np.clip(steer, -params.MAX_STEER_ANGLE, params.MAX_STEER_ANGLE)
        if vx_new > 0.1:
            psi_dot_new = (vx_new / self.L) * math.tan(steer)
        else:
            psi_dot_new = 0.0
        psi_new = psi + psi_dot_new * self.dt

        # position update
        x_new = x + vx_new * math.cos(psi_new) * self.dt
        y_new = y + vx_new * math.sin(psi_new) * self.dt

        return np.array([x_new, y_new, vx_new, 0.0, psi_new, psi_dot_new])

# ============================= Track: projection and interpolation =============================


class Track:
    def __init__(self, waypoints: np.ndarray):
        self.waypoints = waypoints
        self._precompute()

    def _precompute(self):
        self.seg_vectors = np.diff(self.waypoints, axis=0)
        self.seg_lengths = np.linalg.norm(self.seg_vectors, axis=1)
        self.cum_len = np.cumsum(np.insert(self.seg_lengths, 0, 0))
        self.total_length = self.cum_len[-1]

        self.tangents = self.seg_vectors / (self.seg_lengths[:, None] + 1e-8)
        self.normals = np.array([[-t[1], t[0]] for t in self.tangents])
        self.normals = self.normals / (np.linalg.norm(self.normals, axis=1, keepdims=True) + 1e-8)

    def project(self, x: float, y: float) -> Tuple[float, float]:
        """Return (s, e) where e is signed lateral deviation (left positive)"""
        pos = np.array([x, y])
        best_dist = np.inf
        best_idx = -1
        best_t = 0.0
        best_closest = None

        for i, (p0, p1) in enumerate(zip(self.waypoints[:-1], self.waypoints[1:])):
            seg_vec = p1 - p0
            seg_len = self.seg_lengths[i]
            if seg_len < 1e-6:
                continue
            t = np.dot(pos - p0, seg_vec) / (seg_len * seg_len)
            t = np.clip(t, 0.0, 1.0)
            closest = p0 + t * seg_vec
            dist = np.linalg.norm(pos - closest)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
                best_t = t
                best_closest = closest

        normal = self.normals[best_idx]
        signed_e = np.dot(pos - best_closest, normal)
        s = self.cum_len[best_idx] + best_t * self.seg_lengths[best_idx]
        return s, signed_e

    def interpolate(self, s: float) -> Tuple[float, float, float]:
        """Return (x, y, psi) at arc length s"""
        if s <= 0:
            return self.waypoints[0][0], self.waypoints[0][1], self._get_psi_at_seg(0)
        if s >= self.total_length:
            idx = len(self.waypoints) - 2
            return self.waypoints[-1][0], self.waypoints[-1][1], self._get_psi_at_seg(idx)

        idx = np.searchsorted(self.cum_len[1:], s)
        local_s = s - self.cum_len[idx]
        t = local_s / (self.seg_lengths[idx] + 1e-8)
        p0 = self.waypoints[idx]
        p1 = self.waypoints[idx+1]
        x = p0[0] + t * (p1[0] - p0[0])
        y = p0[1] + t * (p1[1] - p0[1])
        psi = self._get_psi_at_seg(idx)
        return x, y, psi

    def _get_psi_at_seg(self, idx: int) -> float:
        if idx >= len(self.tangents):
            idx = len(self.tangents) - 1
        dx, dy = self.tangents[idx]
        return math.atan2(dy, dx)

# ============================= test track generation =============================


def create_test_track(num_points=200, radius=12.0):
    t = np.linspace(0, 2*np.pi, num_points)
    # x = radius * np.cos(t)
    # y = radius * np.sin(t) * 0.6 + 2 * np.sin(2*t)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return np.column_stack([x, y])


# ============================= Control Policies =============================


class ControlPolicy(abc.ABC):
    @abc.abstractmethod
    def get_control_action(self, state: CarState) -> CarAction:
        pass


class DefaultControlPolicy(ControlPolicy):
    """Sinusoidal steering + constant target velocity"""
    def __init__(self, vel: float = 1.0, steer_amplitude: float = 0.5, period: float = 4.0, dt=params.DT):
        self.vel = vel                # target throttle (0~1)
        self.steer_amplitude = steer_amplitude
        self.period = period
        self.time = 0.0
        self.dt = dt

    def get_control_action(self, state: CarState) -> CarAction:
        steer_cmd = self.steer_amplitude * math.sin(2 * math.pi * self.time / self.period)
        self.time += self.dt
        return CarAction(target_vel=self.vel, target_steer=steer_cmd)


class LLMControlPolicy(ControlPolicy):
    """
    LLM-based control (Gemini + LangChain). Optional: requires `langchain-google-genai` and GOOGLE_API_KEY.
    If dependencies not installed, falls back to default policy.
    """
    def __init__(self, model_name="qwen2.5:3b", dt=params.DT):
        self.dt = dt
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.8,
            base_url="http://localhost:11434",
            format="json",
            num_predict=256,
        )
        # self.system_msg = (
        #     "You are a high-precision racing controller. Generate standardized commands based on the vehicle's state.\n"
        #     "You need to finish the track as soon as possible."
        #     "You need to avoid getting out of track which will cause heavy penalty in velocity."
        #     "You are given "
        #     "Output strictly in JSON format: {\"target_vel\": float, \"target_steer\": float}\n"
        #     "target_vel: -1(backwards)~1(forwards); target_steer: -1(left)~1(right)"
        # )
        self.json = json

    def get_control_action(self, state) -> CarAction:
        prompt = (
            f"state: x={state.x:.2f}, y={state.y:.2f}, psi={np.degrees(state.psi):.1f}°, "
            f"vx={state.vx:.2f}, vy={state.vy:.2f}, omega={state.omega:.2f}"
        )
        messages = prompt  # [("system", self.system_msg), ("human", prompt)]
        # print(prompt)
        try:
            response = self.llm.invoke(messages)
            # print(response)
            cmd = self.json.loads(response.content)
            print(cmd)
            return CarAction(
                target_vel=np.clip(float(cmd["target_vel"]), -1, 1),
                target_steer=np.clip(float(cmd["target_steer"]), -1, 1)
            )
        except Exception as e:
            print(f"Error in LLM control: {e}")
            print("Falling back to default policy.")
            return CarAction(target_vel=0.8, target_steer=0.0)

# ============================= Simulation =============================


def run_simulation(controller_type='default', model_name='qwen2.5:3b'):
    waypoints = create_test_track(num_points=300, radius=14.0)
    track = Track(waypoints)

    model = BicycleKinematicModel()

    start_s = 0.0
    x0, y0, psi0 = track.interpolate(start_s)
    # initial state as numpy array: [x, y, vx, vy, psi, psi_dot]
    state = np.array([x0, y0, 0.0, 0.0, psi0, 0.0])

    # Choose policy
    if controller_type == 'default':
      controller = DefaultControlPolicy(vel=0.8, steer_amplitude=0.5, period=6.0, dt=params.DT)
    if controller_type == 'LLM':
      controller = LLMControlPolicy(model_name=model_name)

    total_steps = int(params.TOTAL_TIME / params.DT)
    half_width = params.TRACK_WIDTH / 2.0
    off_thresh = params.OFFROAD_TOLERANCE

    # Data recording
    traj_x, traj_y = [], []
    velocities = []
    lateral_errors = []

    print("Single-vehicle simulation started...")
    for step in range(total_steps):
        t = step * params.DT

        # Convert numpy array to CarState for policy input
        state_obj = CarState.from_array(state)
        action = controller.get_control_action(state_obj)

        # Map action to throttle (0~1) and steering angle (rad)
        throttle = np.clip(action.target_vel, 0.0, 1.0)   # ignore backwards
        steer = np.clip(action.target_steer, -1.0, 1.0) * params.MAX_STEER_ANGLE

        # Step dynamics
        state = model.step(state, throttle, steer)

        # Project to track to get lateral error e
        s, e = track.project(state[0], state[1])
        lateral_errors.append(e)

        # Out-of-track penalty (identical to alpha-RACER)
        if abs(e) > off_thresh:
            overshoot = abs(e) - off_thresh
            factor = np.exp(-params.OFFROAD_STRENGTH * overshoot)
            state[2] *= factor   # vx
            state[3] *= factor   # vy

        # Record every 2 steps
        if step % 2 == 0:
            traj_x.append(state[0])
            traj_y.append(state[1])
            velocities.append(state[2])

        if (step+1) % 100 == 0:
            print(f"Step {step+1}/{total_steps}, t={t:.1f}s, "
                  f"v={state[2]:.2f} m/s, e={e:.3f}")

    print("Simulation finished. Plotting...")

    # ========== Plotting ==========
    plt.figure(figsize=(12, 8))

    # Trajectory
    plt.subplot(2, 2, 1)
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'k--', alpha=0.5, label='Track Center')
    plt.plot(traj_x, traj_y, 'b-', linewidth=1.5, label='Vehicle Trajectory')
    # Mark offroad points where |e| > off_thresh
    off_indices = [i for i, e in enumerate(lateral_errors) if abs(e) > off_thresh]
    if off_indices:
        off_x = [traj_x[i//2] for i in off_indices if i//2 < len(traj_x)]
        off_y = [traj_y[i//2] for i in off_indices if i//2 < len(traj_y)]
        plt.scatter(off_x, off_y, c='red', s=20, alpha=0.6, label='Offroad')
    plt.title('Vehicle Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.legend()

    # Velocity
    plt.subplot(2, 2, 2)
    time_vel = np.arange(len(velocities)) * params.DT * 2
    plt.plot(time_vel, velocities, 'g-')
    plt.title('Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)

    # Lateral error
    plt.subplot(2, 2, 3)
    time_err = np.arange(len(lateral_errors)) * params.DT
    plt.plot(time_err, lateral_errors, 'r-')
    plt.axhline(y=half_width + off_thresh, color='k', linestyle='--', alpha=0.5, label='Offroad threshold')
    plt.axhline(y=-half_width - off_thresh, color='k', linestyle='--', alpha=0.5)
    plt.title('Lateral Error (e)')
    plt.xlabel('Time (s)')
    plt.ylabel('Lateral error (m)')
    plt.legend()
    plt.grid(True)

    # Offroad penalty active indicator
    plt.subplot(2, 2, 4)
    offroad_flag = [1 if abs(e) > off_thresh else 0 for e in lateral_errors]
    time_flag = np.arange(len(offroad_flag)) * params.DT
    plt.plot(time_flag, offroad_flag, drawstyle='steps-post', color='orange')
    plt.title('Offroad Penalty Active')
    plt.xlabel('Time (s)')
    plt.ylabel('Active (0/1)')
    plt.ylim(-0.1, 1.1)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ============================= Main =============================
if __name__ == "__main__":
    run_simulation(controller_type='LLM', model_name="my-racing-ctrl")