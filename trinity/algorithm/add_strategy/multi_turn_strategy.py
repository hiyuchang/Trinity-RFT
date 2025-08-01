# -*- coding: utf-8 -*-
import asyncio
from typing import Dict, List, Tuple

from trinity.algorithm.add_strategy.add_strategy import ADD_STRATEGY, GRPOAddStrategy, group_by
from trinity.buffer import BufferWriter
from trinity.common.experience import Experience
from trinity.utils.monitor import gather_metrics
from trinity.utils.timer import Timer

from trinity.utils.log import get_logger
logger = get_logger(__name__)

@ADD_STRATEGY.register_module("multi_turn")
class MultiTurnAddStrategy(GRPOAddStrategy):
    """An Addstrategy for Reward Propagation Workflow."""
    def __init__(self, writer: BufferWriter, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(writer, epsilon, **kwargs)

    async def add(self, exps: List[Experience], step: int) -> Tuple[int, Dict]:
        if len(exps) == 0:
            return 0, {}

        exp_groups = group_by(exps, id_type="task")
        cnt = 0
        metric_list = []
        metrics = {}
        tasks = []
        
        logger.info(f"debug: begin adding {len(exps)} experiences in step {step}")
        with Timer(metrics, "add_strategy_time"):
            for group_id, group_exps in exp_groups.items():
                
                # --- Debugging Setup ---
                original_steps = [exp.eid.step for exp in group_exps]
                processed_steps = []
                # -----------------------

                exp_by_steps = group_by(group_exps, id_type="step")
                sorted_exp_steps = sorted(exp_by_steps.items(), key=lambda x: x[0], reverse=True)
                
                for step_id, step_exps in sorted_exp_steps:
                    if len(step_exps) < 2:
                        continue
                    
                    logger.info(f"Processing group {group_id} step {step_id} with {len(step_exps)} exps")
                    processed_exps, step_metrics = self.calculate_group_advantage(group_id, step_exps)
                    logger.info(f"{step_metrics=}")
                    metric_list.append(step_metrics)
                    cnt += len(processed_exps)

                    # --- Debugging Setup ---
                    processed_steps.extend([exp.eid.step for exp in processed_exps])
                    # -----------------------

                    if len(processed_exps) > 0:
                        tasks.append(self.writer.write_async(processed_exps))

                logger.info("--- debug ---")
                logger.info(f"Processing Task ID: {group_id}")
                # logger.info(f"{original_steps=}")
                logger.info(f"processed_steps={processed_steps}") 
                logger.info("-------------")
        logger.info(f"debug: finally adding {cnt} experiences in step {step}")
        if tasks:
            await asyncio.gather(*tasks)
        try:
            group_metrics = gather_metrics(metric_list, "group_advantages")
            metrics.update(group_metrics)
        except ValueError:
            pass  # empty metric list causes ValueError, ignore it
            
        return cnt, metrics

    @classmethod
    def default_args(cls) -> dict:
        return {"epsilon": 1e-6,
                "rank_penalty": 0.25}