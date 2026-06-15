from config import params
from main import run_experiment_suite


QWEN_SIZE_CASES = [
    (["LLM", "LLM", "LLM"], 3),
    (["LLM", "LLM", "LLM"], 2),
    (["LLM", "LLM", "LLM"], 1),
]

QWEN_SIZE_MODELS = [
    ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"],
    ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"],
    ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"],
]

DEFAULT_LLM_VS_POTENTIAL_CASES = [
    (["potential_function_policy", "potential_function_policy", "LLM"], 3),
    (["potential_function_policy", "LLM", "potential_function_policy"], 2),
    (["LLM", "potential_function_policy", "potential_function_policy"], 1),
]

DEFAULT_LLM_VS_POTENTIAL_MODELS = [
    [None, None, params.LLM_MODEL],
    [None, params.LLM_MODEL, None],
    [params.LLM_MODEL, None, None],
]


def run_all():
    common_kwargs = dict(
        save_animation=True,
        stride=2,
        interval=30,
        dpi=100,
        show_plot=False,
        show_ref_debug=False,
        vehicle_model_type="kinematic",
        llm_output_mode="mode",
    )

    qwen_dirs = run_experiment_suite(
        start_order_cases=QWEN_SIZE_CASES,
        llm_model_cases=QWEN_SIZE_MODELS,
        name_prefix="qwen-size-competition",
        **common_kwargs,
    )

    default_vs_potential_dirs = run_experiment_suite(
        start_order_cases=DEFAULT_LLM_VS_POTENTIAL_CASES,
        llm_model_cases=DEFAULT_LLM_VS_POTENTIAL_MODELS,
        name_prefix="default-llm-vs-potential",
        **common_kwargs,
    )

    print("\nFinished six experiments:")
    for output_dir in [*qwen_dirs, *default_vs_potential_dirs]:
        print(f"- {output_dir}")


if __name__ == "__main__":
    run_all()
