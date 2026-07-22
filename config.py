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
        self.Ta: float = 5.5    # throttle gain
        self.Tb: float = -1.    # velocity-dependent term
        self.Sa: float = 0.36
        self.Sb: float = 0.03
        self.mu: float = 3.0
        self.Cf: float = 1.0
        self.Cr: float = 1.0
        self.Bf: float = 60.0
        self.Br: float = 60.0
        self.hcom: float = 0.0
        self.fr: float = 0.1    # friction
        self.delay: int = 4
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
    v0 = 0.0
    TRACK_CSV = "track_csv/berlin_2018.csv"
    RACELINE_CSV = "track_csv/berlin_2018_raceline_with_speeds.csv"
    TRACK_SCALE = 2.0
    TRACK_HALF_WIDTH = 0.44
    TRACK_WIDTH = 2.0 * TRACK_HALF_WIDTH
    VEHICLE_HALF_LENGTH = 0.18
    VEHICLE_HALF_WIDTH = 0.12
    # MPC constrains the vehicle center, so reserve one vehicle half-width at
    # each track edge: |e_center| <= track_half_width - vehicle_half_width.
    MPC_CONSTRAINT_HALF_WIDTH = TRACK_HALF_WIDTH - VEHICLE_HALF_WIDTH
    # Keep references slightly inside the hard MPC boundary to leave tracking
    # and numerical feasibility margin instead of targeting the constraint.
    REF_BOUNDARY_MARGIN = 0.03
    REF_LATERAL_CLIP = MPC_CONSTRAINT_HALF_WIDTH - REF_BOUNDARY_MARGIN
    # Smooth collision exclusion used by the continuous NLP solver:
    #
    #     (delta_s / a)^p + (delta_e / b)^p >= 1,
    #
    # where delta_s and delta_e are the longitudinal and lateral separations,
    # and p=4 gives a rounded-rectangle (superellipse).  The paper's literal
    # constraints |delta_s| >= 0.36 AND |delta_e| >= 0.24 would incorrectly
    # prohibit ordinary following and side-by-side racing; collision-free
    # rectangle separation is an OR condition and is awkward for IPOPT.
    #
    # For two axis-aligned cars with combined half-extents ds=0.36 m and
    # de=0.24 m, multiplying both axes by 2**(1/p) makes the p-superellipse
    # contain the complete rectangular overlap region.  With p=4 this gives
    # a ~= 0.428 m and b ~= 0.285 m.  These are MPC centre-to-centre safety
    # separations; track-boundary vehicle-width margins are intentionally not
    # included here.
    MPC_MIN_LONGITUDINAL_SEPARATION = 0.428
    MPC_MIN_LATERAL_SEPARATION = 0.285
    MPC_MAX_CONSTRAINED_OPPONENTS = 2  # nearest car ahead and nearest car behind
    MPC_COLLISION_SUPERELLIPSE_EXPONENT = 4
    OFFROAD_TOLERANCE = 0.05
    OFFROAD_STRENGTH = 3.0 # multiplier for offroad penalty in MPC cost
    MPC_DT = 0.1

    MPC_HORIZON = 10

    Q = np.diag([1.0, 1.0])
    R = np.diag([1.0, 1.0]) #* 1e-6

    VELOCITY_TRACK_WEIGHT = 0.0 # weight for velocity tracking in MPC cost
    HEADING_TRACK_WEIGHT = 0.0 # weight for heading tracking in MPC cost

    MAX_VEL_RATE = 8.0
    MAX_STATE_SPEED = 5.0
    MPC_MAX_STEER_DELTA = 0.4  # max normalized steering-command change per MPC step
    MAX_RACELINE_SPEED = 5.0
    PROJECTION_SEARCH_RADIUS = 1.0
    PROJECTION_MAX_HEADING_ERROR = np.pi / 2.0
    TRACE_OUTPUT = "simulation_trace.npz"
    VEHICLE_MODEL = "kinematic"
    NUM_CARS = 1
    CAR_START_GAP = 6.0
    EARLY_COLLISION_STOP = False # stop early if cars collide immediately after launch
    EARLY_COLLISION_WINDOW_STEPS = 150 # only apply early collision stop during the first N simulation steps
    EARLY_COLLISION_CONSECUTIVE_STEPS = 3 # require this many consecutive collision steps before stopping
    LLM_MODEL = "qwen2.5:0.5b"
    LLM_BASE_URL = "http://localhost:11434"
    LLM_TIMEOUT = 20.0
    LLM_TEMPERATURE = 0.2
    LLM_UPDATE_INTERVAL_STEPS = 50 # how often to update the LLM policy in terms of MPC steps
    LLM_HISTORY_LENGTH = 5 # recent LLM outputs included in the next LLM prompt
    LLM_OUTPUT_MODE = "auto" # "auto" infers from LLM_MODEL; use "mode" or "params" to force a style
    LLM_SAFEGUARD_ENABLED = True # validate/correct discrete LLM modes against the current race context
    # Official alpha-RACER sampling bounds mapped to this project's
    # [q, zeta, s1, s2, s3] parameterization.
    LLM_PARAMETER_BOUNDS = {
        "q": (0.30, 1.00),
        "zeta": (0.85**2, 1.10**2),
        "s1": (0.10, 0.50),
        "s2": (0.10, 0.50),
        "s3": (0.00, 1.00),
    }
    # Previous hand-tuned presets:
    # stable=[1.3,1.0,0.05,1.0,0.0], push=[1.8,1.25,0.05,1.0,0.0]
    # overtake=[1.0,1.35,1.75,0.1,0.0], block=[1.6,0.9,0.05,0.45,2.0]
    # recover=[3.0,0.7,0.0,1.5,0.0]
    LLM_MODE_PRESETS = {
        "stable": np.array([0.75, 1.00, 0.00, 0.00, 0.00]),
        "push": np.array([0.85, 1.20, 0.00, 0.00, 0.00]),
        "overtake": np.array([0.80, 1.35, 0.45, 0.10, 0.00]),
        "block": np.array([0.75, 0.95, 0.00, 0.30, 0.70]),
        "recover": np.array([3.00, 0.75, 0.00, 0.00, 0.00]),
    }
    STATIC_POLICY_PARAMS = np.array([1.0, 1.0, 1.0, 0.5, 0.5])  # q, zeta, s1, s2, s3
    # Upstream optimizer coordinates: sf1, sf2, lookahead_factor,
    # speed_factor, blocking. AlphaRacerPolicy maps these to q, zeta, s1,
    # s2, s3 and then updates them online with the learned potential model.
    ALPHARACER_INITIAL_ORIGINAL_PARAMS = np.array([0.3, 0.2, 0.25, 0.95, 0.0])
    POTENTIAL_FUNCTION_WEIGHT_PATH = "alpha_racer_weights/p_models_rel/model_multi_myopic_mpc.pth"
    POTENTIAL_FUNCTION_OPT_STYLE = "grad"  # Paper/default upstream online optimizer.
    POTENTIAL_FUNCTION_GRADIENT_RATE = 1e-4
    POTENTIAL_FUNCTION_SEARCH_POINTS = 7
    POTENTIAL_FUNCTION_SEARCH_ITERS = 6
    ALPHARACER_WEIGHT_PATH = POTENTIAL_FUNCTION_WEIGHT_PATH  # backward-compatible name
    ALPHARACER_WEIGHT_SEARCH_POINTS = POTENTIAL_FUNCTION_SEARCH_POINTS
    ALPHARACER_WEIGHT_SEARCH_ITERS = POTENTIAL_FUNCTION_SEARCH_ITERS
    LLM_POLICY_PARAMS = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # q, zeta, s1, s2, s3
    SELF_PLAY_RL_MODEL_PATH = "self_play_rl_policy.zip"  # Created by: python self_play_rl.py
    SELF_PLAY_RL_BACKEND = "sb3"  # Paper-compatible Stable-Baselines3 PPO checkpoint.
    SELF_PLAY_RL_STRICT_LOAD = True  # Do not silently replace a missing trained PPO with the heuristic fallback.
    SELF_PLAY_RL_ACTION_ORDER = "throttle_steer"  # use "steer_throttle" for exported policies with reversed outputs
    SELF_PLAY_RL_THROTTLE_BOUNDS = np.array([0.0, 1.0])
    SELF_PLAY_RL_STEER_BOUNDS = np.array([-1.0, 1.0])
    IBR_HORIZON = 10 # Matches multi_car_comp_blocking.py's IBR horizon.
    IBR_ITERATIONS = 3


params = SimParams()
db_params = DynamicParams("dbm")
