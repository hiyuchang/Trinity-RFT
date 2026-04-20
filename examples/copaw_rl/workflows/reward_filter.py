from typing import List, Tuple

import numpy as np

from trinity.buffer.operators import EXPERIENCE_OPERATORS
from trinity.buffer.operators.experience_operator import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.monitor import gather_metrics


@EXPERIENCE_OPERATORS.register_module("reward_std_filter_v2")
class RewardSTDFilterV2(ExperienceOperator):
    """
    Filter experiences based on the standard deviation of rewards within each group.

    Note: This filter assumes that the reward is already calculated and stored in the Experience object.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], dict]:
        """Filter experiences based on reward std."""
        metrics = {}
        result_exps = []
        original_count = len(exps)
        grouped_experiences = group_by(exps, id_type="task")
        metrics_list = []
        for _, group_exps in grouped_experiences.items():
            if len(group_exps) < 2:
                continue
            rewards = [exp.reward for exp in group_exps]
            variance = np.std(rewards)
            metrics_list.append(
                {
                    "reward_mean": np.mean(rewards),
                    "reward_std": variance,
                }
            )
            if variance <= self.threshold:
                continue
            result_exps.extend(group_exps)
        final_count = len(result_exps)
        metrics["operator_filtered_count"] = original_count - final_count
        metrics.update(gather_metrics(metrics_list, "origin_group_advantages"))
        return result_exps, metrics
