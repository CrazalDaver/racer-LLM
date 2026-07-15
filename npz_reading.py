import argparse
from pathlib import Path

import numpy as np


RAW_LLM_FIELDS = (
    "llm_raw_call_indices",
    "llm_raw_prompts",
    "llm_raw_responses",
    "llm_raw_errors",
)


def read_llm_raw_io(npz_path):
    """Print the saved raw input/output for each LLM policy in an NPZ trace."""
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [field for field in RAW_LLM_FIELDS if field not in data.files]
        if missing:
            raise KeyError(
                f"{npz_path} does not contain raw LLM data; missing: "
                + ", ".join(missing)
            )

        call_indices = data["llm_raw_call_indices"]
        prompts = data["llm_raw_prompts"]
        responses = data["llm_raw_responses"]
        errors = data["llm_raw_errors"]
        policy_names = (
            data["policy_names"].astype(str)
            if "policy_names" in data.files
            else np.asarray([""] * len(call_indices))
        )

        for car_idx, calls in enumerate(call_indices):
            valid_slots = [slot for slot, call in enumerate(calls) if int(call) >= 0]
            if not valid_slots:
                continue

            policy_name = policy_names[car_idx] if car_idx < len(policy_names) else ""
            label = f"car {car_idx}"
            if policy_name:
                label += f" ({policy_name})"
            print(f"\n{'=' * 20} {label} {'=' * 20}")

            for slot in valid_slots:
                print(f"\n--- LLM call {int(calls[slot])} ---")
                print("[raw input / prompt]")
                print(str(prompts[car_idx, slot]))
                print("[raw output / response]")
                print(str(responses[car_idx, slot]))
                error = str(errors[car_idx, slot])
                if error:
                    print("[error]")
                    print(error)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read the five most recent raw LLM inputs and outputs from a simulation NPZ."
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        default="simulation_trace.npz",
        help="NPZ trace path (default: simulation_trace.npz)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    read_llm_raw_io(args.npz_path)
