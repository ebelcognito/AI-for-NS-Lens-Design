"""RL environment: the engine as a verifiable reward.

`agent_context/ns_rl_research/18_rl_engine_reward_plan.md` is the plan this implements.
Two things separate this package from `nslens.qd`, and both are deliberate:

  * **The gates are hardened.** Every permissive default in `ExperimentConfig` exists so
    `tests/test_qd_equivalence.py` can compare frozen archives cell-for-cell. Each one is
    also a documented exploit that search already found. See `env_config`.

  * **Gates score, they do not reject.** The QD driver returns None for a gated design;
    a policy-gradient method needs the partial credit, so `reward.evaluate` measures every
    verdict and `reward.reward_of` prices them. Hard zeros are reserved for designs that
    are physically meaningless.
"""
from __future__ import annotations

__all__ = ["data", "env_config", "parse", "prompts", "reward", "server", "tasks"]
