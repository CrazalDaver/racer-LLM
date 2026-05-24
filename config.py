import numpy as np


GRAVITY = 9.81


class DynamicParams:
    def __init__(self):
        self.LF = 0.11
        self.LR = 0.23
        self.MASS = 4.65
        self.Iz = 0.1
        self.Ta = 5.5
        self.Tb = -1.0
        self.Sa = 0.36
        self.Sb = 0.03
        self.mu = 3.0
        self.Cf = 1.0
        self.Cr = 1.0
        self.Bf = 60.0
        self.Br = 60.0
        self.hcom = 0.0
        self.fr = 0.1


class SimParams:
    DT = 0.02
    TOTAL_TIME = 100.0
    TRACK_CSV = "track_csv/berlin_2018.csv"
    RACELINE_CSV = "track_csv/berlin_2018_raceline_with_speeds.csv"
    TRACK_WIDTH = 2.0
    OFFROAD_TOLERANCE = 0.55
    OFFROAD_STRENGTH = 3.0
    MPC_HORIZON = 6
    Q = np.diag([5.0, 5.0, 0.5, 1.0])
    R = np.diag([0.05, 0.005])
    MAX_VEL_RATE = 8.0
    MAX_STATE_VEL_RATE = 4.0
    MAX_STATE_SPEED = 6.0
    MAX_LATERAL_SPEED = 3.0
    MAX_YAW_RATE = 8.0
    MAX_STEER_RATE = 4.0
    MAX_RACELINE_SPEED = 3.0
    PROJECTION_SEARCH_RADIUS = 4.0
    RECOVERY_ERROR_THRESHOLD = 0.65
    RECOVERY_THROTTLE = 0.35


params = SimParams()
db_params = DynamicParams()
