"""Run alpha-RACER's numerical race without installing ROS2.

This adapter replaces only ROS timers, publishers and message containers.
The upstream race callback, dynamics, controllers, weights and initial states
are imported and executed directly from the unmodified sibling repository.
"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import types

import numpy as np
import torch


class _Message:
    def __init__(self):
        self.header = types.SimpleNamespace(frame_id="", stamp=None)
        point = lambda: types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        quat = lambda: types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)
        pose = lambda: types.SimpleNamespace(position=point(), orientation=quat())
        direct_pose = pose()
        direct_pose.pose = pose()
        self.pose = direct_pose
        self.twist = types.SimpleNamespace(
            twist=types.SimpleNamespace(linear=point(), angular=point())
        )
        self.polygon = types.SimpleNamespace(points=[])
        self.poses = []
        self.markers = []
        self.data = 0.0


class _Publisher:
    def publish(self, _message):
        pass


class _Clock:
    def now(self):
        return types.SimpleNamespace(to_msg=lambda: None)


class _Node:
    def __init__(self, _name):
        self._timers = []

    def create_publisher(self, *_args, **_kwargs):
        return _Publisher()

    def create_timer(self, period, callback):
        self._timers.append((float(period), callback))
        return callback

    def get_clock(self):
        return _Clock()

    def destroy_node(self):
        pass


def _install_ros_stubs(steps):
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda: None
    rclpy.shutdown = lambda: None

    def spin(node):
        callback = min(node._timers, key=lambda item: item[0])[1]
        globals_ = callback.__globals__
        generators = [
            globals_["waypoint_generator"],
            globals_["waypoint_generator_opp"],
            globals_["waypoint_generator_opp1"],
        ]
        latest_lateral = [np.nan, np.nan, np.nan]
        for idx, generator in enumerate(generators):
            original_generate = generator.generate

            def recorded_generate(*args, _idx=idx, _generate=original_generate, **kwargs):
                result = _generate(*args, **kwargs)
                if len(result) >= 4:
                    latest_lateral[_idx] = float(result[3])
                return result

            generator.generate = recorded_generate
        lateral_history = [[], [], []]
        progress_history = [[], [], []]
        for _ in range(steps):
            callback()
            for idx in range(3):
                lateral_history[idx].append(latest_lateral[idx])
            for idx, key in enumerate(("s", "s_opp", "s_opp1")):
                progress_history[idx].append(float(globals_.get(key, np.nan)))
        summary = {
            "steps": steps,
            "dt": float(globals_.get("DT", np.nan)),
            "final_progress": [values[-1] for values in progress_history],
            "max_abs_lateral_error": [
                float(np.nanmax(np.abs(values))) for values in lateral_history
            ],
            "offroad_steps_gt_055": [
                int(np.count_nonzero(np.abs(values) > 0.55))
                for values in lateral_history
            ],
            "final_lateral_error": [values[-1] for values in lateral_history],
        }
        print("ALPHA_HEADLESS_SUMMARY=" + json.dumps(summary))

    rclpy.spin = spin
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_node.Node = _Node
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = rclpy_node

    message_modules = {
        "geometry_msgs.msg": [
            "PoseWithCovarianceStamped", "PoseStamped", "Point",
            "PolygonStamped", "Point32", "TwistStamped",
        ],
        "nav_msgs.msg": ["Path", "Odometry"],
        "visualization_msgs.msg": ["MarkerArray", "Marker"],
        "std_msgs.msg": ["Float64", "Int8"],
        "ackermann_msgs.msg": ["AckermannDrive"],
    }
    for module_name, class_names in message_modules.items():
        package_name = module_name.split(".")[0]
        sys.modules.setdefault(package_name, types.ModuleType(package_name))
        module = types.ModuleType(module_name)
        for class_name in class_names:
            setattr(module, class_name, _Message)
        sys.modules[module_name] = module

    transforms = types.ModuleType("tf_transformations")
    transforms.quaternion_from_euler = lambda _r, _p, yaw: np.array(
        [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)]
    )
    transforms.euler_matrix = lambda _r, _p, yaw: np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0, 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    sys.modules["tf_transformations"] = transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, default=Path("../alpha-RACER"))
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--opp1", default="mpc")
    parser.add_argument("--opp2", default="mpc")
    parser.add_argument("--mpc", action="store_true")
    parser.add_argument("--rel", action="store_true")
    args = parser.parse_args()

    root = args.upstream_root.resolve()
    race_dir = root / "car_ros2" / "car_ros2"
    script = race_dir / "multi_car_comp_blocking.py"
    if not script.exists():
        raise FileNotFoundError(script)
    _install_ros_stubs(args.steps)
    # Released checkpoints were serialized from CUDA.  Mapping storage to CPU
    # changes neither tensors nor model equations and is required on this Mac.
    upstream_torch_load = torch.load
    torch.load = lambda *load_args, **load_kwargs: upstream_torch_load(
        *load_args, map_location="cpu", **load_kwargs
    )
    sys.path[:0] = [str(race_dir), str(root / "car_dynamics")]
    upstream_argv = [str(script), "--opp1", args.opp1, "--opp2", args.opp2]
    if args.mpc:
        upstream_argv.append("--mpc")
    if args.rel:
        upstream_argv.append("--rel")
    sys.argv = upstream_argv
    os.chdir(race_dir)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
