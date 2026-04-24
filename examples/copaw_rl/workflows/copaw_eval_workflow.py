from typing import List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task


@WORKFLOWS.register_module("copaw_eval_workflow")
class CoPawEvalWorkflow(MultiTurnWorkflow):
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
            run_eval_workflow,
        )

        sandbox_id = self.task.workflow_args.get("sandbox_id", None)
        token = self.task.workflow_args["token"]
        domain = self.task.workflow_args["domain"]
        template = self.task.workflow_args["template"]
        checkpoint_job_dir = self.task.workflow_args["checkpoint_job_dir"]

        sandbox, created = get_or_create_sandbox(sandbox_id, token, domain, template, self.logger)

        oss_config = self.task.workflow_args["oss"]
        dashscope_api_key = self.task.workflow_args["dashscope_api_key"]
        task_id = self.task.raw_task["task_id"]
        api_server_url = f"{self.model.api_address}/v1"
        model_path = self.model.model_path
        model_version = self.model.model_version
        try:
            metrics = run_eval_workflow(
                sandbox,
                task_id,
                oss_config,
                dashscope_api_key,
                api_server_url,
                model_path,
                f"step_{model_version}",
                checkpoint_job_dir,
                self.logger,
            )
            metrics = {
                key: value for key, value in metrics.items() if isinstance(value, (int, float))
            }
        except Exception as e:
            self.logger.error(f"Error running workflow (ID: {sandbox.sandbox_id}): {e}")
            raise e
        finally:
            sandbox.kill()

        exps = [
            Experience(
                tokens=torch.tensor([]),
                logprobs=torch.tensor([]),
                prompt_length=0,
                action_mask=torch.tensor([]),
                reward=0.0,
                metrics=metrics,
            )
        ]

        self.logger.info(
            f"Workflow finished. Sandbox {'created' if created else 'connected'} "
            f"(ID: {sandbox.sandbox_id}). Collected {len(exps)} experiences."
        )

        return exps
