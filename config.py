"""Repository-wide defaults and vehicle parameter presets.

Runtime overrides belong in ``simulation_core.config.SimulationConfig``.
This module contains shared defaults used when an override is absent.
"""

import copy

import numpy as np


GRAVITY = 9.81

# Canonical high-level policy coordinates used throughout this repository.
# They match the alpha-RACER paper: theta=(q, zeta, s1, s2, s3).
ALPHARACER_PARAMETER_NAMES = ("q", "zeta", "s1", "s2", "s3")
ALPHARACER_PARAMETER_BOUNDS = np.asarray(
    [
        [1.0, 1.0 / (2.0 * 0.15)],  # q; converted from upstream lookahead [0.15, 0.50]
        [0.85**2, 1.10**2],         # zeta; converted from upstream speed factor
        [0.10, 0.50],               # s1
        [0.10, 0.50],               # s2
        [0.00, 1.00],               # s3
    ],
    dtype=float,
)


class DynamicParams:
    """Physical and actuator parameters for one vehicle model."""

    def __init__(self, vehicle_model_type="dbm"):
        model_type = vehicle_model_type.lower()

        # Geometry and mass
        self.LF: float = 0.11
        self.LR: float = 0.23
        self.MASS: float = 4.65
        self.Iz: float = 0.1

        # Integration and tire model
        self.DT: float = 0.05
        self.K_RFY: float = 20.0
        self.K_FFY: float = 20.0
        self.mu: float = 3.0
        self.Cf: float = 1.0
        self.Cr: float = 1.0
        self.Bf: float = 60.0
        self.Br: float = 60.0
        self.hcom: float = 0.0
        self.fr: float = 0.1

        # Throttle and steering actuators
        self.Ta: float = 5.5
        self.Tb: float = -1.0
        self.Sa: float = 0.36
        self.Sb: float = 0.03
        self.delay: int = 4
        self.ros_kinematic = False

        if model_type in {"kinematic", "kinetic", "kinetics"}:
            self._use_kinematic_baseline()
        elif model_type == "dbm":
            self._use_upstream_numerical_dbm()
        else:
            raise ValueError(f"Unknown vehicle model: {vehicle_model_type}")

    def _use_upstream_numerical_dbm(self):
        """Match alpha-RACER's numerical-race DBM instantiation.

        ``multi_car_comp_blocking.py`` overrides the model class defaults with
        Sa=0.34, Sb=0, Ta=20, Tb=0, mu=0.5 and delay=1.  Keeping these values
        explicit prevents a DBM ablation from silently using the unrelated
        library defaults above.  The local simulator integrates this plant at
        SIM_DT; the upstream experiment used DT=0.1 s.
        """
        self.LF = 0.11
        self.LR = 0.23
        self.MASS = 4.65
        self.Iz = 0.1
        self.Ta = 20.0
        self.Tb = 0.0
        self.Sa = 0.34
        self.Sb = 0.0
        self.mu = 0.5
        self.Cf = 1.0
        self.Cr = 1.0
        self.Bf = 60.0
        self.Br = 60.0
        self.hcom = 0.0
        self.fr = 0.1
        self.delay = 1

    def _use_kinematic_baseline(self):
        # Match the released alpha-RACER ROS MPC prediction model.  Earlier
        # local revisions used Sa=0.80 as a raceline-tracking workaround,
        # which more than doubled the modeled steering authority.
        self.LF = 0.11
        self.LR = 0.23
        self.Sa = 0.34
        self.Sb = 0.0
        self.Ta = 20.0 / 4.65
        self.Tb = 0.0
        self.fr = 1.0
        self.ros_kinematic = True

    def copy(self):
        return copy.copy(self)


def make_vehicle_params(vehicle_model_type):
    return DynamicParams(vehicle_model_type)


def duration_to_steps(duration_seconds, dt):
    """Convert a positive duration to steps without triggering it early."""
    return max(
        1,
        int(np.ceil(float(duration_seconds) / float(dt) - 1e-12)),
    )


class SimParams:
    # Frozen experiment contract. Any change to its plant, MPC, recovery,
    # track/raceline, timing, parameter space, scenarios, or metrics requires
    # a new version; see protocols/K-Constrained-v1.json.
    PROTOCOL_VERSION = "K-Constrained-v1"

    # ==================== SIMULATION TIMING ====================

    # The simulator, policies, and MPC all advance on the upstream 0.1 s step.
    # LLM policies may still update their high-level parameters less often.
    SIM_DT = 0.1
    CONTROL_DT = 0.1
    TOTAL_TIME = 50.0
    v0 = 0.0

    # ==================== TRACK AND START GRID =================

    TRACK_CSV = "track_csv/berlin_2018.csv"
    # Locally recomputed minimum-curvature raceline (Sa = 0.34).
    RACELINE_CSV = "track_csv/generated/berlin_mincurv_iqp_local.csv"
    TRACK_SCALE = 2.0
    TRACK_HALF_WIDTH = 0.44
    TRACK_WIDTH = 2.0 * TRACK_HALF_WIDTH
    NUM_CARS = 1
    CAR_START_GAP = 6.0

    # ==================== VEHICLE AND CONTACT ==================

    VEHICLE_MODEL = "kinematic"
    VEHICLE_HALF_LENGTH = 0.18
    VEHICLE_HALF_WIDTH = 0.12
    # Upstream-style asymmetric overlap decay applied once per 0.1 s step.
    COLLISION_FRONT_DECAY = 5.0
    COLLISION_REAR_DECAY = 20.0

    # ==================== MPC SAFETY CONSTRAINTS ===============

    # Boundary handling treats each car as a point: both the reference and
    # predicted vehicle centre may use the full geometric track half-width.
    # Vehicle dimensions remain relevant only to contact detection/visuals.
    MPC_CONSTRAINT_HALF_WIDTH = TRACK_HALF_WIDTH
    REF_BOUNDARY_MARGIN = 0.0
    REF_LATERAL_CLIP = TRACK_HALF_WIDTH - REF_BOUNDARY_MARGIN
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
    # Paper constraints (1f)-(1h) are softened with non-negative violation
    # variables. Large quadratic weights preserve safety in normal operation
    # while allowing IPOPT to recover from an already unsafe measured state.
    MPC_ENFORCE_TRACK_BOUNDARY = True
    MPC_ENFORCE_COLLISION_AVOIDANCE = True
    # Track slack is a lateral distance in metres. Collision slack is instead
    # the dimensionless residual of the normalized superellipse constraint,
    # so their weights are intentionally independent rather than numerically
    # tied. In the rho_t=1e3 experiments the unchanged collision weight is
    # therefore deliberately larger, prioritizing avoidance of vehicle overlap.
    # A lower track penalty was more tolerant of the temporary boundary
    # violation needed to escape the Berlin hairpin's low-speed equilibrium.
    # Collision slack remains more expensive because it is dimensionless and
    # represents vehicle overlap rather than lateral boundary overshoot.
    MPC_TRACK_SLACK_WEIGHT = 1.0e3
    MPC_COLLISION_SLACK_WEIGHT = 1.0e4
    # Released alpha-RACER MPC constraint:
    # |v^2 tan(steer_command)| <= mu * g * wheelbase.
    # It progressively limits steering at high speed using a friction proxy.
    MPC_ENFORCE_FRICTION_SPEED_LIMIT = True
    MPC_FRICTION_COEFFICIENT = 1.6

    # ==================== RECOVERY AND LIVENESS ================

    # Numerical-simulator recovery threshold.  This is deliberately separate
    # from TRACK_HALF_WIDTH: the alpha-RACER paper defines recovery outside the
    # track boundary, while the released Berlin simulation uses |e| > 0.55 m.
    # Keeping the implementation threshold explicit avoids presenting the
    # difference as a geometric "tolerance" derived by the model.
    OFFROAD_RECOVERY_THRESHOLD = 0.55
    # Released alpha-RACER numerical-simulation recovery. For overshoot d, the
    # upstream code applies once per 0.1 s step:
    #   v *= exp(-3 d)
    #   psi += (1 - exp(-d)) * heading_error
    #   steer += (-sign(e) - steer) * (1 - exp(-3 d))
    # and adds 0.2 throttle when |heading_error| > 1 rad.
    OFFROAD_STRENGTH = 3.0
    OFFROAD_HEADING_REALIGN_STRENGTH = 1.0
    OFFROAD_INWARD_STEER = 1.0
    OFFROAD_LARGE_HEADING_ERROR = 1.0
    OFFROAD_THROTTLE_BOOST = 0.2
    # Kinematic-model compatibility: below this speed, force at least the
    # neutral throttle plus the upstream +0.2 boost so an aligned car cannot
    # remain trapped at the zero-acceleration equilibrium.
    OFFROAD_LOW_SPEED_THRESHOLD = 0.3
    # ==================== MPC CONTROL AND COST =================

    MPC_DT = 0.1
    # Optional detachable comparison profile.  ``constrained_local`` remains
    # the default and preserves all existing experiments.
    MPC_FORMULATION = "constrained_local"

    MPC_HORIZON = 10

    Q = np.diag([1.0, 1.0])
    R = np.diag([1.0, 1.0]) #* 1e-6

    # Shared reference-speed tracking term.  It helps prevent the zero-speed
    # receding-horizon equilibrium observed around the Berlin hairpin.
    VELOCITY_TRACK_WEIGHT = 10.0
    HEADING_TRACK_WEIGHT = 0.0 # weight for heading tracking in MPC cost
    MAX_VEL_RATE = 8.0
    MAX_STATE_SPEED = 5.0
    MPC_MAX_STEER_RATE = 4.0  # normalized steering-command change per second (0.4 per 0.1 s MPC step)
    MAX_RACELINE_SPEED = 5.0
    PROJECTION_SEARCH_RADIUS = 1.0
    PROJECTION_MAX_HEADING_ERROR = np.pi / 2.0

    # ==================== SCENARIO AND OUTPUT ==================

    TRACE_OUTPUT = "simulation_trace.npz"
    REF_DEBUG_SAMPLE_INTERVAL_SECONDS = 2.0
    TRAJECTORY_SAMPLE_INTERVAL_SECONDS = 0.1
    PROGRESS_REPORT_INTERVAL_SECONDS = 4.0

    # ==================== LLM POLICIES =========================

    LLM_MODEL = "qwen2.5:7b"
    LLM_BASE_URL = "http://localhost:11434"
    LLM_TIMEOUT = 20.0
    LLM_TEMPERATURE = 0.0
    LLM_UPDATE_INTERVAL_SECONDS = 1.0
    # Direct-parameter input modules. The base always contains ego_state,
    # every opponent in traffic_state, and road_preview. Non-base modules are
    # independent switches so later ablations change one factor at a time.
    LLM_INPUT_INCLUDE_RACELINE_SPEED = False
    LLM_INPUT_INCLUDE_CURRENT_PARAMETERS = False
    LLM_INPUT_INCLUDE_DECISION_FEEDBACK = False
    # Maximum number of executed modes exposed in
    # memory.recent_modes_oldest_to_newest.
    LLM_RECENT_MODES_MAX_LENGTH = 1
    LLM_OUTPUT_MODE = "auto" # "auto" infers from LLM_MODEL; use "mode" or "params" to force a style
    # Observation-only tolerance for longitudinal relative speed. Values inside
    # this band are reported as a constant gap and never produce a closure time.
    LLM_RELATIVE_SPEED_DEADBAND_MPS = 0.1
    # LLM scene classification uses a small buffer around the MPC collision
    # envelope so millimetre-scale threshold crossings do not imply clear road.
    LLM_CRITICAL_DISTANCE = 1.0
    LLM_CLOSE_DISTANCE = 2.5
    LLM_OVERTAKE_DISTANCE = 4.0
    LLM_BLOCK_MAX_TTC_SECONDS = 1.0
    LLM_SAME_LANE_MARGIN = 0.05
    LLM_PUSH_MAX_ABS_CURVATURE = 0.15
    # Ego-centered centerline/raceline preview exposed to the LLM. Sampling
    # resolution is observational only; it does not classify the road or pick
    # a behavior mode.
    LLM_ROAD_PREVIEW_LENGTH_METERS = 8.0 
    LLM_ROAD_PREVIEW_SAMPLE_COUNT = 5

    # ==================== ALPHARACER POLICIES ==================

    # Official alpha-RACER sampling bounds mapped to this project's
    # [q, zeta, s1, s2, s3] parameterization. The upstream
    # lookahead_factor range [0.15, 0.50] maps through
    # q = 1 / (2 * lookahead_factor).
    ALPHARACER_PARAMETER_NAMES = ALPHARACER_PARAMETER_NAMES
    ALPHARACER_PARAMETER_BOUNDS = ALPHARACER_PARAMETER_BOUNDS
    LLM_PARAMETER_BOUNDS = {
        name: tuple(float(value) for value in ALPHARACER_PARAMETER_BOUNDS[idx])
        for idx, name in enumerate(ALPHARACER_PARAMETER_NAMES)
    }
    # Discrete modes within LLM_PARAMETER_BOUNDS. Push keeps fast reference
    # progression; overtake reduces it while applying a broad lateral offset.
    # Stable/push use high s2 to localize the unavoidable minimum s1 response.
    LLM_MODE_PRESETS = {
        "stable": np.array([1.0 / 0.75, 1.00, 0.10, 0.50, 0.00]),
        "push": np.array([1.0 / 0.85, 1.20, 0.10, 0.50, 0.00]),
        "overtake": np.array([1.0 / 0.80, 1.20, 0.45, 0.10, 0.00]),
        "block": np.array([1.0 / 0.75, 1.0, 0.10, 0.30, 0.70]),
    }
    # Receding-horizon oracle defaults.  The oracle deliberately enumerates
    # LLM_MODE_PRESETS at runtime instead of maintaining a second mode list.
    # Its real-race dwell matches the current LLM cadence.  Three seconds
    # matches the existing fixed-mode diagnostic horizon and is long enough
    # for short overtaking/blocking effects while remaining configurable.
    ORACLE_HORIZON_SECONDS = 3.0
    ORACLE_COMMITMENT_SECONDS = LLM_UPDATE_INTERVAL_SECONDS
    # The repository's primary race metric is progress.  Safety and relative
    # progress remain explicit optional score terms rather than hidden reward
    # shaping; every raw component is logged even when its weight is zero.
    ORACLE_PROGRESS_WEIGHT = 1.0
    ORACLE_RELATIVE_PROGRESS_WEIGHT = 0.0
    ORACLE_COLLISION_PENALTY = 0.0
    ORACLE_OFFTRACK_PENALTY_PER_SECOND = 0.0
    ORACLE_TIE_TOLERANCE = 1.0e-9
    # Conservative fixed baseline within the upstream parameter ranges:
    # moderate tracking, slightly slower reference progression, small
    # overtaking displacement, and no blocking.
    STATIC_POLICY_PARAMS = np.array([1.0 / 0.75, 0.90, 0.20, 0.30, 0.00])  # q, zeta, s1, s2, s3
    # Canonical paper coordinates: q, zeta, s1, s2, s3. Official checkpoints
    # are adapted internally because they were trained in transformed coords.
    ALPHARACER_INITIAL_PARAMS = np.array([2.0, 0.95**2, 0.3, 0.2, 0.0])
    POTENTIAL_FUNCTION_WEIGHT_PATH = (
        "policy/alpharacer/train/weights/upstream/potential_myopic_mpc.pth"
    )
    POTENTIAL_FUNCTION_OPT_STYLE = "grad"
    # Step size in normalized paper-parameter coordinates xi in [0,1]^15.
    POTENTIAL_FUNCTION_GRADIENT_RATE = 1e-4
    POTENTIAL_FUNCTION_SEARCH_POINTS = 7
    POTENTIAL_FUNCTION_SEARCH_ITERS = 6
    ALPHARACER_WEIGHT_PATH = POTENTIAL_FUNCTION_WEIGHT_PATH  # backward-compatible name
    ALPHARACER_WEIGHT_SEARCH_POINTS = POTENTIAL_FUNCTION_SEARCH_POINTS
    ALPHARACER_WEIGHT_SEARCH_ITERS = POTENTIAL_FUNCTION_SEARCH_ITERS
    # Valid neutral starting point in canonical paper coordinates.  Match the
    # local alpha-RACER initializer so direct-parameter LLM experiments do not
    # begin from out-of-range s1/s2 values before the first successful call.
    LLM_POLICY_PARAMS = np.array([2.0, 0.95**2, 0.30, 0.20, 0.00])

    # ==================== SELF-PLAY RL =========================

    # Created by: python -m policy.self_play_rl.train
    SELF_PLAY_RL_MODEL_PATH = "policy/self_play_rl/weights/self_play_rl_policy.zip"
    SELF_PLAY_RL_BACKEND = "sb3"  # Paper-compatible Stable-Baselines3 PPO checkpoint.
    SELF_PLAY_RL_STRICT_LOAD = True  # Do not silently replace a missing trained PPO with the heuristic fallback.
    SELF_PLAY_RL_ACTION_ORDER = "throttle_steer"  # use "steer_throttle" for exported policies with reversed outputs
    SELF_PLAY_RL_THROTTLE_BOUNDS = np.array([0.0, 1.0])
    SELF_PLAY_RL_STEER_BOUNDS = np.array([-1.0, 1.0])

    # ==================== TRAJECTORY IBR =======================

    # IBR_IMPLEMENTATION choices:
    #   local_wang      Wang-style IBR using this project's dynamics,
    #                   constraints, action definition, and MPC timing.
    #   historical_strict restores the local strict solver used from commit
    #                   5ab4f74 onward: upstream objective plus local vehicle
    #                   parameters, hard track bounds, and own-action recovery.
    IBR_IMPLEMENTATION = "local_wang"
    # local_wang uses complete ego/opponent rounds and finishes with a final
    # ego solve.
    IBR_HORIZON = 10
    IBR_ITERATIONS = 2
    IBR_PROGRESS_WEIGHT = 10.0
    # IBR objective audit (2026-08): progress is the Wang-style racing
    # objective. Across six scenarios, the released alpha-RACER IBR's 10*e^4
    # lateral form eliminated recorded off-road time without materially
    # reducing progress. Heading and speed tracking were not needed.
    IBR_LATERAL_WEIGHT = 10.0
    IBR_LATERAL_EXPONENT = 4
    IBR_HEADING_WEIGHT = 0.0
    # Compatibility default for trace readers. Local trajectory IBR uses the
    # effective shared MPC velocity weight so CLI overrides remain aligned.
    IBR_SPEED_WEIGHT = VELOCITY_TRACK_WEIGHT
    IBR_CONTROL_WEIGHT = 0.0
    # ``neutral`` penalizes deviation from straight, speed-holding commands;
    # ``zero`` reproduces the released alpha-RACER IBR's ||u||^2 form.
    IBR_CONTROL_REFERENCE = "neutral"  # inactive while IBR_CONTROL_WEIGHT == 0
    IBR_CONTROL_RATE_WEIGHT = 0.10
    # Main baseline uses the shared global 5 m/s state bound. ``raceline`` is
    # an ablation targeting the historical 1.04 * local-raceline-speed cap;
    # its next-step bound is relaxed only as much as physical braking requires.
    IBR_SPEED_CAP_MODE = "global"  # "global" or "raceline"
    IBR_RACELINE_SPEED_CAP_FACTOR = 1.04
    # Wang et al. use hard constraints. In this discrete three-car simulator,
    # full-run ablations showed that hard track constraints enter a recovery
    # infeasibility loop and hard pairwise collision constraints fail during a
    # closing pack. Keep both feasibility slacks explicit and measurable.
    IBR_ALLOW_TRACK_SLACK = True
    IBR_ALLOW_COLLISION_SLACK = True
    IBR_CONVERGENCE_TOLERANCE = 1.0e-3
    # Historical local port kept three alternating solves and the upstream
    # friction proxy while taking vehicle geometry/actuation from local MPC.
    IBR_HISTORICAL_ITERATIONS = 3
    IBR_HISTORICAL_FRICTION_COEFFICIENT = 1.6

    # ==================== PARAMETER IBR ========================

    # Five-parameter IBR diagnostic. Each best response searches canonical
    # coordinates (q, zeta, s1, s2, s3), while candidate actions and
    # trajectories are always produced by the shared alpha-RACER/LLM MPC.
    # This is intentionally slower than the learned potential policy and is
    # primarily an offline/reference baseline rather than a real-time method.
    PARAMETER_IBR_ROUNDS = 2
    PARAMETER_IBR_SEARCH_POINTS = 5
    PARAMETER_IBR_SEARCH_PASSES = 2
    PARAMETER_IBR_UPDATE_INTERVAL_SECONDS = 1.0
    PARAMETER_IBR_CONVERGENCE_TOLERANCE = 1.0e-3


params = SimParams()
db_params = DynamicParams("dbm")
