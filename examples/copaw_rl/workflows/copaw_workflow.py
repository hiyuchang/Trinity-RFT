from typing import Any, List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.rewards.open_judge_reward import TrajectoryAccuracyGrader
from trinity.common.rewards.reward_fn import RewardFn
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task


@WORKFLOWS.register_module("copaw_workflow")
class CoPawWorkflow(MultiTurnWorkflow):
    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        reward_fn_args = task.reward_fn_args or {}
        if isinstance(task.reward_fn, type) and issubclass(task.reward_fn, RewardFn):
            self.reward_fn: RewardFn = task.reward_fn(**reward_fn_args)
        else:
            # Fallback to OpenJudge trajectory grader when reward_fn is not explicitly configured.
            self.reward_fn = TrajectoryAccuracyGrader(**reward_fn_args)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            text_parts = []
            for item in response:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join([p for p in text_parts if p])
        return ""

    @classmethod
    def _build_openjudge_inputs(cls, trajectories: Any) -> tuple[Experience, List[dict]]:
        if not isinstance(trajectories, list) or not trajectories:
            exp = Experience(tokens=torch.tensor([0, 1]), prompt_length=1, response_text="")
            return exp, []

        last_traj = trajectories[-1] if isinstance(trajectories[-1], dict) else {}
        messages = last_traj.get("messages", [])
        if not isinstance(messages, list):
            messages = []

        response_text = cls._extract_response_text(last_traj.get("response"))
        exp = Experience(tokens=torch.tensor([0, 1]), prompt_length=1, response_text=response_text)
        return exp, messages

    def run(self):
        from examples.copaw_rl.workflows.sandbox_utils import (
            get_or_create_sandbox,
            run_workflow,
        )

        sandbox_id = self.task.workflow_args.get("sandbox_id", None)
        token = self.task.workflow_args["token"]
        domain = self.task.workflow_args["domain"]
        template = self.task.workflow_args["template"]

        sandbox, created = get_or_create_sandbox(sandbox_id, token, domain, template, self.logger)
        self.logger.info(f"Sandbox {sandbox.sandbox_id} created: {created}")

        oss_config = self.task.workflow_args["oss"]
        dashscope_api_key = self.task.workflow_args["dashscope_api_key"]
        task_id = self.task.raw_task["task_id"]
        api_server_url = f"{self.model.api_address}/v1"
        model_path = self.model.model_path
        try:
            dataset = run_workflow(
                sandbox,
                task_id,
                oss_config,
                dashscope_api_key,
                api_server_url,
                model_path,
                self.logger,
            )
        except Exception as e:
            self.logger.error(f"Error running workflow (ID: {sandbox.sandbox_id}): {e}")
            sandbox.kill()
            raise e
        exps = []
        
        trajectories = dataset[-1].pop("trajectories")
        session = dataset[-1].pop("session", None)
        judge_exp, judge_messages = self._build_openjudge_inputs(trajectories)
        reward_dict = self.reward_fn(judge_exp, judge_messages, session=session)  # type: ignore[misc]
        reward = reward_dict.get("reward", sum(reward_dict.values()))

        for data in dataset:
            prompt_token_ids = torch.tensor(data["prompt_token_ids"])
            response_token_ids = torch.tensor(data["token_ids"])
            token_ids = torch.cat([prompt_token_ids, response_token_ids])
            logprobs = torch.tensor(data["logprobs"])
            prompt_length = len(prompt_token_ids)
            action_mask = torch.tensor(data["response_mask"], dtype=torch.int)
            # reward = float(data.get("judge_ok", 0.0))
            exp = Experience(
                tokens=token_ids,
                logprobs=logprobs,
                prompt_length=prompt_length,
                action_mask=action_mask,
                reward=reward,
            )
            exps.append(exp)

        # sandbox.kill()
        self.logger.info(
            f"Workflow finished. Sandbox {'created' if created else 'connected'} "
            f"(ID: {sandbox.sandbox_id}). Collected {len(exps)} experiences."
        )

        return exps
