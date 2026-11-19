"""Official alpha-RACER three-car race (release 9cff403, car_ros2/multi_car_comp_blocking.py, run with
--mpc --rel --opt_style grad) without ROS, reorganized as world / controller / policies / session.
Numerics are the official code: vendored car_dynamics (jax DBM, WaypointGenerator, OffroadCar),
mpc_controller.py, model_arch.py, and the CarNode methods and mpc_ibr copied verbatim
(controller.py, ibr.py). Run with the official environment (jax 0.4.30, torch 2.8.0, casadi 3.8.2)."""
import os
import sys

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR not in sys.path:
    sys.path.append(_VENDOR)  # vendored car_dynamics uses absolute imports; appended so repo modules win
