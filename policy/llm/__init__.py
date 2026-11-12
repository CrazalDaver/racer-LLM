from .base import LLMPolicyBase, resolve_llm_output
from .hybrid import LLMModeAlphaRacerPolicy
from .mode import LLMModePolicy
from .parameter import LLMParameterPolicy


class LLMAlphaRacerPolicy:
    """Construct the concrete mode or parameter policy for the legacy API.

    Input is the former ``output_mode`` plus the normal policy arguments. The
    result is an ``LLMModePolicy`` or ``LLMParameterPolicy`` instance; new code
    should instantiate those concrete classes directly.
    """

    def __new__(
        cls,
        *args,
        output_mode="auto",
        model_name="qwen2.5:7b",
        param_bounds=None,
        include_raceline_speed=False,
        include_current_parameters=False,
        include_decision_feedback=False,
        **kwargs,
    ):
        """Resolve the old combined constructor to one concrete subclass."""
        response_format = resolve_llm_output(output_mode, model_name)
        if response_format == "mode":
            if any(
                (
                    include_raceline_speed,
                    include_current_parameters,
                    include_decision_feedback,
                )
            ):
                raise ValueError(
                    "Direct-parameter input-module flags require output_mode='params'"
                )
            policy_class = LLMModePolicy
        else:
            policy_class = LLMParameterPolicy
            kwargs.update(
                param_bounds=param_bounds,
                include_raceline_speed=include_raceline_speed,
                include_current_parameters=include_current_parameters,
                include_decision_feedback=include_decision_feedback,
            )
        return policy_class(*args, model_name=model_name, **kwargs)


__all__ = [
    "LLMAlphaRacerPolicy",
    "LLMPolicyBase",
    "LLMModeAlphaRacerPolicy",
    "LLMModePolicy",
    "LLMParameterPolicy",
    "resolve_llm_output",
]
