"""Detachable raw-prompt ablation that removes the implied default mode.

The policy inherits the complete raw LLM implementation and appends exactly
one neutral instruction. It does not prescribe a mode mapping or usage quota.
"""

from .llm import LLMAlphaRacerPolicy


class BalancedRawLLMPolicy(LLMAlphaRacerPolicy):
    """Raw mode policy with a single no-default-mode prompt intervention."""

    PROMPT_ADDITION = (
        "Treat the four modes as situation-dependent tactical tools. No mode is "
        "the default: maximizing race progress may require controlled following, "
        "overtaking, or blocking rather than always selecting push."
    )

    def __init__(self, *args, **kwargs):
        kwargs["output_mode"] = "mode"
        kwargs["prompt_variant"] = "experimental_raw_no_default_mode"
        super().__init__(*args, **kwargs)
        self.system_prompt = f"{self.system_prompt.rstrip()}\n\n{self.PROMPT_ADDITION}"
