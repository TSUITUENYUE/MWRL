"""verl agent loop that ends each rollout at the closing answer tag.

A base model given the chat template answers and then keeps generating unrelated text, often
with more <answer> tags, until <|endoftext|>. This loop is verl's single-turn loop with the
stop string ``</answer>`` (kept in the output), so a rollout is exactly the reasoning plus one
answer. ``evaluate.py`` samples with the same stop string. Imported by
``verl_ext.register`` inside every Ray worker; select it with
``actor_rollout_ref.rollout.agent.default_agent_loop=ds_answer_stop``.
"""

from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

STOP = ["</answer>"]


@register("ds_answer_stop")
class AnswerStopAgentLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params: dict[str, Any], **kwargs):
        params = {**sampling_params, "stop": STOP, "include_stop_str_in_output": True}
        return await super().run(params, **kwargs)
