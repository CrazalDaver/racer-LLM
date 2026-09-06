"""Detachable two-call variant of the evidence-first LLM policy.

Stage 1 extracts one observable traffic relation. Stage 2 receives the exact
same observation plus that relation and selects a mode. Everything else
(features, presets, MPC, scheduling, fallback, and trace fields) is inherited.
Remove this file and its small registration entries to remove the experiment.
"""

from __future__ import annotations

import json

from .llm import EVIDENCE_FIRST_BASIS_LABELS, EvidenceFirstLLMPolicy
from .experimental_balanced_raw_llm import BalancedRawLLMPolicy


class TwoStepEvidenceLLMPolicy(EvidenceFirstLLMPolicy):
    """Separate observable evidence extraction from tactical mode selection."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_variant = "experimental_two_step_evidence"
        # Keep the common task, observation, and mode descriptions byte-for-byte
        # identical to the one-call evidence-first prompt. Only its output
        # procedure is replaced below.
        marker = "Use the following output process:"
        common, separator, _ = self.system_prompt.partition(marker)
        if not separator:
            raise ValueError(f"Evidence-first prompt is missing marker: {marker}")
        self._two_step_common_system = common.rstrip()

    @property
    def _basis_system_prompt(self):
        return self._two_step_common_system + """

For this first stage, identify only the single observable longitudinal traffic
relationship most relevant to the current decision. Do not select or recommend
a racing mode. Return only the JSON required by the response schema.
"""

    @property
    def _mode_system_prompt(self):
        return self._two_step_common_system + """

For this second stage, the user message supplies the same numerical observation
and one traffic basis produced by a separate first stage. Use the complete
observation and the supplied basis to select one racing mode. There is no fixed
basis-to-mode mapping. Return only the JSON required by the response schema.
"""

    @staticmethod
    def _basis_schema():
        return {
            "type": "object",
            "properties": {
                "basis": {
                    "type": "string",
                    "enum": list(EVIDENCE_FIRST_BASIS_LABELS),
                }
            },
            "required": ["basis"],
            "additionalProperties": False,
        }

    def _mode_schema(self):
        return {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": list(self.mode_presets)},
                "reason": {"type": "string", "maxLength": 120},
            },
            "required": ["mode", "reason"],
            "additionalProperties": False,
        }

    def _query_llm(self, prompt: str) -> dict:
        basis_result = super()._query_llm(
            prompt,
            system_prompt=self._basis_system_prompt,
            output_schema=self._basis_schema(),
        )
        stage_one = self._query_metadata(basis_result)
        basis = str(basis_result["basis"])

        # Preserve the one-call observation layout and add exactly one field;
        # nesting the observation would itself be an avoidable prompt confound.
        mode_payload = json.loads(prompt)
        mode_payload["provided_basis"] = basis
        mode_prompt = json.dumps(mode_payload, indent=2, ensure_ascii=True)
        mode_result = super()._query_llm(
            mode_prompt,
            system_prompt=self._mode_system_prompt,
            output_schema=self._mode_schema(),
        )
        stage_two = self._query_metadata(mode_result)

        # Preserve the established trace format while retaining both raw calls
        # in one JSON response for later latency and reasoning audits.
        self._last_raw_llm_response = json.dumps(
            {"basis_stage": stage_one, "mode_stage": stage_two}, ensure_ascii=True
        )
        self._last_llm_latency_seconds = (
            stage_one["latency_seconds"] + stage_two["latency_seconds"]
        )
        self._last_llm_eval_count = self._sum_counts(
            stage_one["eval_count"], stage_two["eval_count"]
        )
        self._last_llm_prompt_eval_count = self._sum_counts(
            stage_one["prompt_eval_count"], stage_two["prompt_eval_count"]
        )
        self._last_llm_done_reason = f"basis:{stage_one['done_reason']};mode:{stage_two['done_reason']}"
        return {
            "basis": basis,
            "mode": mode_result["mode"],
            "reason": mode_result["reason"],
        }

    def _query_metadata(self, parsed):
        return {
            "parsed": parsed,
            "raw_response": self._last_raw_llm_response,
            "done_reason": self._last_llm_done_reason,
            "eval_count": self._last_llm_eval_count,
            "prompt_eval_count": self._last_llm_prompt_eval_count,
            "latency_seconds": float(self._last_llm_latency_seconds),
        }

    @staticmethod
    def _sum_counts(first, second):
        return first + second if first >= 0 and second >= 0 else -1


class BalancedTwoStepEvidenceLLMPolicy(TwoStepEvidenceLLMPolicy):
    """Two-step evidence policy with the same no-default mode intervention.

    The basis stage is deliberately unchanged. The balanced sentence is added
    only to the tactical mode stage, so it cannot directly bias which factual
    relationship the first call extracts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_variant = "experimental_balanced_two_step_evidence"

    @property
    def _mode_system_prompt(self):
        return (
            super()._mode_system_prompt
            + "\n\n"
            + BalancedRawLLMPolicy.PROMPT_ADDITION
        )
