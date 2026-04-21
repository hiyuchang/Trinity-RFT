from typing import Any, List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
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
        for data in dataset:
            prompt_token_ids = torch.tensor(data["prompt_token_ids"])
            response_token_ids = torch.tensor(data["token_ids"])
            token_ids = torch.cat([prompt_token_ids, response_token_ids])
            logprobs = torch.tensor(data["logprobs"])
            prompt_length = len(prompt_token_ids)
            action_mask = torch.tensor(data["response_mask"], dtype=torch.int)
            reward = float(data.get("reward", 0.0))
            exp = Experience(
                tokens=token_ids,
                logprobs=logprobs,
                prompt_length=prompt_length,
                action_mask=action_mask,
                reward=reward,
            )
            exps.append(exp)

        sandbox.kill()
        self.logger.info(
            f"Workflow finished. Sandbox {'created' if created else 'connected'} "
            f"(ID: {sandbox.sandbox_id}). Collected {len(exps)} experiences."
        )

        return exps
