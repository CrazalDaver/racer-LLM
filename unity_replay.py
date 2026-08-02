"""Render saved three-car races with the bundled Unity replay application.

The script converts an NPZ (or an embedded JSON trial) to a replay file,
launches Unity, captures frames, and produces a GIF or MP4.

Examples:
    python unity_replay.py simulation_trace.npz
    python unity_replay.py result/position_sweep_summary.json --trial 2
    python unity_replay.py simulation_trace.npz --unity-build /path/to/env-v0.app
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_FIELDS = ("time", "car_x", "car_y", "car_psi")
ACTION_FIELDS_PER_CAR = 3
DEFAULT_CAR_COUNT = 3
DEFAULT_UNITY_BUILD = Path(__file__).resolve().parent / "unity_build" / "AlphaRacerReplay.app"


@dataclass
class RaceTrace:
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    source: str
    throttle: np.ndarray | None = None
    track_csv: str = ""
    raceline_csv: str = ""
    track_scale: float = 1.0
    track_width: float = 1.0
    vehicle_half_length: float = 0.18
    vehicle_half_width: float = 0.12
    policies: tuple[str, ...] = ()

    @property
    def car_count(self) -> int:
        return int(self.x.shape[1])

    @property
    def duration(self) -> float:
        return float(self.time[-1] - self.time[0]) if len(self.time) > 1 else 0.0


def _json_array(value):
    """Convert JSON values to arrays, restoring non-finite values saved as null."""
    if value is None:
        return np.asarray(np.nan)
    if isinstance(value, list):
        value = [_replace_null(item) for item in value]
    return np.asarray(value)


def _replace_null(value):
    if value is None:
        return np.nan
    if isinstance(value, list):
        return [_replace_null(item) for item in value]
    return value


def _scalar_string(mapping, name, default="") -> str:
    if name not in mapping:
        return default
    value = np.asarray(mapping[name])
    return str(value.item()) if value.shape == () else str(value)


def _strings(mapping, name) -> tuple[str, ...]:
    if name not in mapping:
        return ()
    value = np.asarray(mapping[name])
    if value.shape == ():
        return (str(value.item()),)
    return tuple(str(item) for item in value.tolist())


def _from_mapping(mapping, source: str) -> RaceTrace:
    missing = [name for name in REQUIRED_FIELDS if name not in mapping]
    if missing:
        raise ValueError(
            f"{source} has no replayable trajectory; missing: {', '.join(missing)}"
        )

    arrays = {name: np.asarray(mapping[name], dtype=float) for name in REQUIRED_FIELDS}
    trace = RaceTrace(
        time=arrays["time"],
        x=arrays["car_x"],
        y=arrays["car_y"],
        yaw=arrays["car_psi"],
        source=source,
        throttle=(
            np.asarray(mapping["car_throttle"], dtype=float)
            if "car_throttle" in mapping else None
        ),
        track_csv=_scalar_string(mapping, "track_csv"),
        raceline_csv=_scalar_string(mapping, "raceline_csv"),
        track_scale=float(np.asarray(mapping["track_scale"]).item()) if "track_scale" in mapping else 1.0,
        track_width=float(np.asarray(mapping["track_width"]).item()) if "track_width" in mapping else 1.0,
        vehicle_half_length=(
            float(np.asarray(mapping["vehicle_half_length"]).item())
            if "vehicle_half_length" in mapping else 0.18
        ),
        vehicle_half_width=(
            float(np.asarray(mapping["vehicle_half_width"]).item())
            if "vehicle_half_width" in mapping else 0.12
        ),
        policies=_strings(mapping, "policy_names"),
    )
    validate_trace(trace)
    return trace


def load_trace(path: Path, trial: int = 1) -> RaceTrace:
    path = path.resolve()
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return _from_mapping(data, str(path))
    if path.suffix.lower() != ".json":
        raise ValueError("Input must be an .npz trace or multi-trial .json summary")

    summary = json.loads(path.read_text(encoding="utf-8"))
    trials = summary.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"{path} does not contain a non-empty trials list")
    if trial < 1 or trial > len(trials):
        raise ValueError(f"Trial must be in 1..{len(trials)}, got {trial}")
    record = trials[trial - 1]
    embedded = record.get("npz")
    if not isinstance(embedded, dict):
        raise ValueError(
            f"{path} trial {trial} only contains aggregate results. "
            "It needs an embedded 'npz' object with time/car_x/car_y/car_psi."
        )
    mapping = {name: _json_array(value) for name, value in embedded.items()}
    if "policy_names" not in mapping and "policy_names" in summary:
        mapping["policy_names"] = np.asarray(summary["policy_names"])
    return _from_mapping(mapping, f"{path} trial {trial}")


def validate_trace(trace: RaceTrace) -> None:
    if trace.time.ndim != 1 or not len(trace.time):
        raise ValueError("time must be a non-empty one-dimensional array")
    expected = (len(trace.time), DEFAULT_CAR_COUNT)
    for name, values in (("car_x", trace.x), ("car_y", trace.y), ("car_psi", trace.yaw)):
        if values.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinite values")
    if trace.throttle is not None:
        if trace.throttle.shape != expected:
            raise ValueError(
                f"car_throttle must have shape {expected}, got {trace.throttle.shape}"
            )
        if not np.all(np.isfinite(trace.throttle)):
            raise ValueError("car_throttle contains NaN or infinite values")
    if not np.all(np.isfinite(trace.time)):
        raise ValueError("time contains NaN or infinite values")
    if len(trace.time) > 1 and np.any(np.diff(trace.time) <= 0.0):
        raise ValueError("time must be strictly increasing")


def transform_trace(
    trace: RaceTrace,
    fps: float,
    scale: float,
    offset_x: float,
    offset_y: float,
    yaw_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    output_time = np.arange(0.0, trace.duration + 0.5 / fps, 1.0 / fps)
    sample_time = np.minimum(output_time, trace.duration)
    source_time = trace.time - trace.time[0]
    output = np.empty((len(output_time), trace.car_count * ACTION_FIELDS_PER_CAR))
    throttle = np.zeros((len(output_time), trace.car_count), dtype=float)
    for car in range(trace.car_count):
        output[:, 3 * car] = np.interp(sample_time, source_time, trace.x[:, car]) * scale + offset_x
        output[:, 3 * car + 1] = np.interp(sample_time, source_time, trace.y[:, car]) * scale + offset_y
        # Unwrap first so interpolation across +/-pi takes the short rotation.
        unwrapped = np.unwrap(trace.yaw[:, car])
        output[:, 3 * car + 2] = np.interp(sample_time, source_time, unwrapped) + yaw_offset
        if trace.throttle is not None:
            throttle[:, car] = np.interp(sample_time, source_time, trace.throttle[:, car])
    return output_time, output, throttle


def print_report(trace: RaceTrace, output_time: np.ndarray, actions: np.ndarray) -> None:
    source_hz = 1.0 / np.median(np.diff(trace.time)) if len(trace.time) > 1 else 0.0
    print("Unity replay compatibility: OK")
    print(f"source: {trace.source}")
    print(f"frames/cars/duration: {len(trace.time)} / {trace.car_count} / {trace.duration:.2f}s")
    print(f"output duration: {output_time[-1]:.2f}s (recorded time)")
    print(f"source rate -> replay rate: {source_hz:.2f} Hz -> {1 / np.median(np.diff(output_time)):.2f} Hz")
    print(f"track: {trace.track_csv or 'not recorded'}")
    print(f"policies: {', '.join(trace.policies) or 'not recorded'}")
    print(
        "mapped bounds: "
        f"x=[{actions[:, 0::3].min():.3f}, {actions[:, 0::3].max():.3f}], "
        f"y=[{actions[:, 1::3].min():.3f}, {actions[:, 1::3].max():.3f}]"
    )


def export_replay(path: Path, output_time: np.ndarray, actions: np.ndarray, throttle: np.ndarray, trace: RaceTrace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        time=output_time,
        unity_actions=actions,
        unity_throttle=throttle,
        source=np.asarray(trace.source),
        track_csv=np.asarray(trace.track_csv),
        policy_names=np.asarray(trace.policies),
    )
    print(f"Exported {len(output_time)} Unity frames to {path}")


def _load_track(trace: RaceTrace, position_scale: float) -> np.ndarray:
    path = Path(trace.track_csv)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Recorded track CSV not found: {path}")
    points = np.loadtxt(path, delimiter=",", comments="#", usecols=(0, 1), dtype=float)
    return points * trace.track_scale * position_scale


def _load_raceline(trace: RaceTrace, position_scale: float) -> np.ndarray:
    if not trace.raceline_csv:
        return np.empty((0, 2), dtype=float)
    path = Path(trace.raceline_csv)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.exists():
        raise FileNotFoundError(f"Recorded raceline CSV not found: {path}")
    points = np.loadtxt(path, delimiter=",", comments="#", usecols=(1, 2), dtype=float)
    return points * trace.track_scale * position_scale


def _write_unity_json(
    path: Path, trace: RaceTrace, output_time: np.ndarray, actions: np.ndarray,
    throttle: np.ndarray, scale: float, visual_track_half_width: float
) -> None:
    track = _load_track(trace, scale)
    raceline = _load_raceline(trace, scale)
    payload = {
        "fps": float(1.0 / np.median(np.diff(output_time))) if len(output_time) > 1 else 30.0,
        "trackWidth": float(2.0 * visual_track_half_width * scale),
        "vehicleLength": float(2.0 * trace.vehicle_half_length * scale),
        "vehicleWidth": float(2.0 * trace.vehicle_half_width * scale),
        "trackX": track[:, 0].tolist(),
        "trackY": track[:, 1].tolist(),
        "racelineX": raceline[:, 0].tolist(),
        "racelineY": raceline[:, 1].tolist(),
        "frames": [
            {"poses": pose.tolist(), "throttle": pedal.tolist()}
            for pose, pedal in zip(actions, throttle)
        ],
    }
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _app_executable(build: Path) -> Path:
    if build.suffix == ".app":
        executable = build / "Contents" / "MacOS" / build.stem
        if executable.exists():
            return executable
        candidates = list((build / "Contents" / "MacOS").glob("*"))
        if candidates:
            return candidates[0]
    return build


def _make_gif(frames_dir: Path, output: Path, fps: float) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("GIF creation requires Pillow: pip install Pillow") from exc
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"Unity produced no PNG frames in {frames_dir}")
    images = [Image.open(path).convert("RGB") for path in frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output, save_all=True, append_images=images[1:], duration=round(1000.0 / fps), loop=0
    )
    for image in images:
        image.close()


def _burn_overlays(
    frames_dir: Path, fps: float, track: np.ndarray, actions: np.ndarray
) -> None:
    """Burn deterministic timer and track overview into off-screen frames."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Timer overlay requires Pillow: pip install Pillow") from exc
    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24
        )
    except OSError:
        font = ImageFont.load_default()
    map_left, map_top, map_width, map_height = 18, 18, 230, 155
    padding = 14
    minimum = track.min(axis=0)
    span = np.maximum(track.max(axis=0) - minimum, 1e-6)
    map_scale = min(
        (map_width - 2 * padding) / span[0],
        (map_height - 2 * padding) / span[1],
    )

    def map_point(x, y):
        used_w, used_h = span * map_scale
        offset_x = map_left + (map_width - used_w) / 2
        offset_y = map_top + (map_height - used_h) / 2
        return (
            round(offset_x + (x - minimum[0]) * map_scale),
            round(offset_y + used_h - (y - minimum[1]) * map_scale),
        )

    track_points = [map_point(x, y) for x, y in track]
    car_colors = ((70, 200, 255), (255, 105, 65), (255, 220, 55))
    for frame_index, path in enumerate(sorted(frames_dir.glob("frame_*.png"))):
        seconds = frame_index / fps
        minutes = int(seconds // 60)
        text = f"TIME  {minutes:02d}:{seconds - 60 * minutes:05.2f}"
        with Image.open(path) as source:
            image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rounded_rectangle(
                (map_left, map_top, map_left + map_width, map_top + map_height),
                radius=10, fill=(15, 20, 26), outline=(230, 235, 240), width=2,
            )
            draw.line(track_points + [track_points[0]], fill=(30, 35, 40), width=8, joint="curve")
            draw.line(track_points + [track_points[0]], fill=(205, 210, 215), width=3, joint="curve")
            pose = actions[min(frame_index, len(actions) - 1)]
            for car, color in enumerate(car_colors):
                cx, cy = map_point(pose[3 * car], pose[3 * car + 1])
                draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=color, outline="black", width=2)
            box = draw.textbbox((0, 0), text, font=font, stroke_width=2)
            x = image.width - (box[2] - box[0]) - 24
            draw.text(
                (x, 22), text, font=font, fill="white",
                stroke_width=2, stroke_fill=(20, 20, 20),
            )
            image.save(path)


def _write_engine_audio(path: Path, throttle: np.ndarray, fps: float) -> None:
    """Synthesize a lightweight three-car engine bed from recorded throttle."""
    sample_rate = 44100
    duration = len(throttle) / fps
    sample_count = max(1, round(duration * sample_rate))
    sample_time = np.arange(sample_count, dtype=float) / sample_rate
    frame_time = np.arange(len(throttle), dtype=float) / fps
    stereo = np.zeros((sample_count, 2), dtype=float)
    pans = (-0.35, 0.0, 0.35)
    base_frequencies = (76.0, 68.0, 72.0)
    for car in range(throttle.shape[1]):
        pedal = np.clip(np.interp(sample_time, frame_time, throttle[:, car]), -1.0, 1.0)
        effort = np.clip(np.abs(pedal), 0.0, 1.0)
        frequency = base_frequencies[car] + 125.0 * effort
        phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
        tone = np.sin(phase) + 0.32 * np.sin(2.0 * phase) + 0.12 * np.sin(3.0 * phase)
        tone *= 0.025 + 0.055 * effort
        left = np.sqrt((1.0 - pans[car]) * 0.5)
        right = np.sqrt((1.0 + pans[car]) * 0.5)
        stereo[:, 0] += tone * left
        stereo[:, 1] += tone * right
    fade = min(sample_count // 2, round(sample_rate * 0.08))
    if fade:
        envelope = np.ones(sample_count)
        envelope[:fade] = np.linspace(0.0, 1.0, fade)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade)
        stereo *= envelope[:, None]
    pcm = np.asarray(np.clip(stereo, -0.95, 0.95) * 32767, dtype="<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def replay(
    build: Path,
    trace: RaceTrace,
    output_time: np.ndarray,
    actions: np.ndarray,
    throttle: np.ndarray,
    scale: float,
    visual_track_half_width: float,
    output: Path,
    keep_frames: bool,
) -> None:
    if visual_track_half_width <= 0.0:
        raise ValueError("visual track half-width must be positive")
    build = build.resolve()
    if not build.exists():
        raise FileNotFoundError(
            f"Unity build not found: {build}. Build/copy AlphaRacerReplay.app there first."
        )
    output = output.resolve()
    work = output.parent / (output.stem + "_unity_frames")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    replay_json = work / "replay.json"
    _write_unity_json(
        replay_json, trace, output_time, actions, throttle, scale,
        visual_track_half_width,
    )
    executable = _app_executable(build)
    unity_log = work / "unity_player.log"
    player_arguments = [
        "-batchmode",
        "-logFile", str(unity_log),
        "--replay", str(replay_json), "--frames", str(work),
        "-screen-width", "960", "-screen-height", "540",
    ]
    print("Starting Unity renderer in background...")
    result = subprocess.run(
        [str(executable), *player_arguments], check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Unity renderer exited with code {result.returncode}; inspect {unity_log}"
        )
    if not (work / "COMPLETE").exists():
        raise RuntimeError("Unity exited before completing frame capture")
    fps = float(1.0 / np.median(np.diff(output_time))) if len(output_time) > 1 else 30.0
    _burn_overlays(work, fps, _load_track(trace, scale), actions)
    if output.suffix.lower() == ".mp4":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg
                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError as exc:
                raise RuntimeError(
                    "MP4 output needs FFmpeg. Install it with: pip install imageio-ffmpeg"
                ) from exc
        engine_audio = work / "engine.wav"
        _write_engine_audio(engine_audio, throttle, fps)
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                "-i", str(work / "frame_%06d.png"), "-i", str(engine_audio),
                "-c:v", "libx264", "-c:a", "aac", "-b:a", "160k", "-shortest",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
            ],
            check=True,
        )
    elif output.suffix.lower() == ".gif":
        _make_gif(work, output, fps)
    else:
        raise ValueError("--output must end in .gif or .mp4")
    if not keep_frames:
        shutil.rmtree(work)
    print(f"Rendered video: {output}")


def open_local_video(output: Path) -> None:
    """Open a rendered MP4 with the operating system's local media player."""
    if sys.platform == "darwin" and output.suffix.lower() == ".mp4":
        result = subprocess.run(["open", str(output.resolve())], check=False)
        if result.returncode != 0:
            print(f"warning: could not open local video: {output}", file=sys.stderr)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="single-run NPZ or multi-trial JSON")
    parser.add_argument("--trial", type=int, default=1, help="1-based JSON trial number")
    parser.add_argument("--export", type=Path, help="write transformed Unity actions to NPZ")
    parser.add_argument(
        "--unity-build",
        type=Path,
        default=DEFAULT_UNITY_BUILD,
        help=f"compatible Unity executable/app (default: {DEFAULT_UNITY_BUILD})",
    )
    parser.add_argument("--output", type=Path, help="rendered .gif or .mp4 path")
    parser.add_argument("--keep-frames", action="store_true", help="keep captured PNG frames")
    parser.add_argument(
        "--no-open", action="store_true",
        help="do not open the completed MP4 in the local macOS player",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="rendered video frame rate")
    parser.add_argument("--scale", type=float, default=1.0, help="world-to-Unity position scale")
    parser.add_argument(
        "--visual-track-half-width", type=float, default=0.55,
        help="Unity asphalt half-width only; does not change simulation constraints",
    )
    parser.add_argument("--offset-x", type=float, default=0.0)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--yaw-offset", type=float, default=0.0, help="Unity heading offset in radians")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        trace = load_trace(args.input, args.trial)
        output_time, actions, throttle = transform_trace(
            trace, args.fps, args.scale, args.offset_x, args.offset_y, args.yaw_offset
        )
        print_report(trace, output_time, actions)
        print(
            f"visual asphalt: half-width {args.visual_track_half_width:.3f} m / "
            f"full width {2 * args.visual_track_half_width:.3f} m"
        )
        if args.export:
            export_replay(args.export, output_time, actions, throttle, trace)
        if not args.export or args.unity_build.exists():
            output = args.output or args.input.with_name(args.input.stem + "_unity.mp4")
            replay(
                args.unity_build, trace, output_time, actions, throttle,
                args.scale, args.visual_track_half_width, output, args.keep_frames
            )
            if not args.no_open:
                open_local_video(output)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
