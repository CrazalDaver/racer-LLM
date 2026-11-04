"""Exact, independent snapshots of the mutable race runtime."""

from __future__ import annotations

import pickle

import casadi as cs


def discard_branch_log(*_args, **_kwargs):
    """Silence expected solver recovery messages in hypothetical branches."""


class RuntimeSnapshotter:
    """Serialize environment, policies, and MPC solvers without touching physics.

    CasADi solver objects are not isolated by ``copy.deepcopy``.  Its global
    pickle contexts provide independent solver instances for every restore.
    Presentation-only logger callbacks are removed only while serializing.
    """

    def __init__(self, race):
        self.race = race

    def capture(self) -> bytes:
        saved_loggers = [policy.log_fn for policy in self.race.policies]
        try:
            for policy in self.race.policies:
                policy.log_fn = print
            with cs.global_pickle_context():
                return pickle.dumps(
                    (
                        self.race.environment,
                        self.race.policies,
                        self.race.mpcs,
                    ),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as exc:
            raise RuntimeError("Could not serialize the simulator runtime") from exc
        finally:
            for policy, logger in zip(self.race.policies, saved_loggers):
                policy.log_fn = logger

    @staticmethod
    def restore(snapshot: bytes):
        try:
            with cs.global_unpickle_context():
                environment, policies, mpcs = pickle.loads(snapshot)
        except Exception as exc:
            raise RuntimeError(
                "Could not independently restore a simulator branch"
            ) from exc
        for policy in policies:
            policy.log_fn = discard_branch_log
        return environment, policies, mpcs
