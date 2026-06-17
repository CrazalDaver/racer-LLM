import numpy as np
import copy


GRAVITY = 9.81


class DynamicParams:
    def __init__(self, vehicle_model_type="dbm"):
        model_type = vehicle_model_type.lower()
        self.LF: float = .11
        self.LR: float = .23
        self.MASS: float = 4.65
        self.DT: float = .05
        self.K_RFY: float = 20.
        self.K_FFY: float = 20.
        self.Iz: float = 0.1
        self.Ta: float = 5.5
        self.Tb: float = -1.
        self.Sa: float = 0.36
        self.Sb: float = 0.03
        self.mu: float = 3.0
        self.Cf: float = 1.0
        self.Cr: float = 1.0
        self.Bf: float = 60.0
        self.Br: float = 60.0
        self.hcom: float = 0.0
        self.fr: float = 0.1
        self.delay: int = 4
        # self.LF = 0.11
        # self.LR = 0.23
        # self.MASS = 4.65 # kg
        # self.Iz = 0.1
        # self.Ta = 8.0    # throttle gain
        # self.Tb = -1.0   # velocity-dependent term
        # self.Sa = 0.60   # 
        # self.Sb = 0.0
        # self.mu = 3.0
        # self.Cf = 1.0
        # self.Cr = 1.0
        # self.Bf = 60.0
        # self.Br = 60.0
        # self.hcom = 0.0
        # self.fr = 0.1    # friction
        self.ros_kinematic = False

        if model_type in {"kinematic", "kinetic", "kinetics"}:
            self._use_kinematic_baseline()
        elif model_type != "dbm":
            raise ValueError(f"Unknown vehicle model: {vehicle_model_type}")

    def _use_kinematic_baseline(self):
        # Match the alpha-RACER ROS kinematic baseline while using the larger
        # steering scale needed by the current raceline curvature.
        self.LF = 0.11
        self.LR = 0.23
        self.Sa = 0.80 # 
        self.Sb = 0.0
        self.Ta = 20.0 / 4.65
        self.Tb = 0.0
        self.fr = 1.0
        self.ros_kinematic = True

    def copy(self):
        return copy.copy(self)


def make_vehicle_params(vehicle_model_type):
    return DynamicParams(vehicle_model_type)


class SimParams:
    DT = 0.02
    TOTAL_TIME = 50.0
    TRACK_CSV = "track_csv/berlin_2018.csv"
    RACELINE_CSV = "track_csv/berlin_2018_raceline_with_speeds.csv"
    TRACK_SCALE = 2.0
    TRACK_HALF_WIDTH = 0.44
    TRACK_WIDTH = 2.0 * TRACK_HALF_WIDTH
    MPC_BOUNDARY_SLACK = 0.0
    OFFROAD_TOLERANCE = 0.0
    OFFROAD_STRENGTH = 3.0 # multiplier for offroad penalty in MPC cost
    MPC_DT = 0.1

    MPC_HORIZON = 8

    Q = np.diag([1.0, 1.0])
    R = np.diag([1.0, 1.0]) * 1e-6

    VELOCITY_TRACK_WEIGHT = 0.0 # weight for velocity tracking in MPC cost
    HEADING_TRACK_WEIGHT = 0.0 # weight for heading tracking in MPC cost

    MAX_VEL_RATE = 8.0
    MAX_STATE_SPEED = 5.0
    MAX_STEER_RATE = 4.0
    MAX_RACELINE_SPEED = 5.0
    PROJECTION_SEARCH_RADIUS = 1.0
    PROJECTION_MAX_HEADING_ERROR = np.pi / 2.0
    TRACE_OUTPUT = "simulation_trace.npz"
    VEHICLE_MODEL = "kinematic"
    NUM_CARS = 1
    CAR_START_GAP = 3.0
    UNSAFE_DISTANCE = 0.4
    EARLY_COLLISION_STOP = False # stop early if cars collide immediately after launch
    EARLY_COLLISION_WINDOW_STEPS = 150 # only apply early collision stop during the first N simulation steps
    EARLY_COLLISION_CONSECUTIVE_STEPS = 3 # require this many consecutive collision steps before stopping
    LLM_MODEL = "my-racing-ctrl-mode"
    LLM_BASE_URL = "http://localhost:11434"
    LLM_TIMEOUT = 20.0
    LLM_UPDATE_INTERVAL_STEPS = 5 # how often to update the LLM policy in terms of MPC steps
    LLM_HISTORY_LENGTH = 5 # recent LLM outputs included in the next LLM prompt
    LLM_OUTPUT_MODE = "auto" # "auto" infers from LLM_MODEL; use "mode" or "params" to force a style
    STATIC_POLICY_PARAMS = np.array([1.0, 1.0, 0.0, 0.0, 0.0])  # q, zeta, s1, s2, s3
    ALPHARACER_PARAMS = STATIC_POLICY_PARAMS  # backward-compatible name
    POTENTIAL_FUNCTION_WEIGHT_PATH = "alpha_racer_weights/p_models_rel/model_multi_myopic_mpc.pth"
    POTENTIAL_FUNCTION_Q = 1.0
    POTENTIAL_FUNCTION_SEARCH_POINTS = 7
    POTENTIAL_FUNCTION_SEARCH_ITERS = 1
    ALPHARACER_WEIGHT_PATH = POTENTIAL_FUNCTION_WEIGHT_PATH  # backward-compatible name
    ALPHARACER_WEIGHT_Q = POTENTIAL_FUNCTION_Q
    ALPHARACER_WEIGHT_SEARCH_POINTS = POTENTIAL_FUNCTION_SEARCH_POINTS
    ALPHARACER_WEIGHT_SEARCH_ITERS = POTENTIAL_FUNCTION_SEARCH_ITERS
    LLM_POLICY_PARAMS = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # q, zeta, s1, s2, s3
    SELF_PLAY_RL_MODEL_PATH = None  # .npz/.json local policy, or SB3 .zip if stable-baselines3 is installed
    SELF_PLAY_RL_BACKEND = "auto"  # auto, npz, json, npy, sb3/ppo
    SELF_PLAY_RL_STRICT_LOAD = False  # False falls back to a simple direct policy when the model is unavailable
    SELF_PLAY_RL_ACTION_ORDER = "throttle_steer"  # use "steer_throttle" for exported policies with reversed outputs
    SELF_PLAY_RL_THROTTLE_BOUNDS = np.array([0.0, 1.0])
    SELF_PLAY_RL_STEER_BOUNDS = np.array([-1.0, 1.0])
    IBR_HORIZON = 10 # Matches multi_car_comp_blocking.py's IBR horizon.
    IBR_ITERATIONS = 3
    IBR_GAMMA = 0.99
    IBR_Q = 1.2
    IBR_ZETA = 1.0
    IBR_S1 = 0.2
    IBR_S2 = 0.2
    IBR_S3 = 0.2
    IBR_PARAMS = np.array([IBR_Q, IBR_ZETA, IBR_S1, IBR_S2, IBR_S3])


params = SimParams()
db_params = DynamicParams("dbm")
