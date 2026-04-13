from typing import List, Optional

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
        url = self.task.workflow_args["url"]
        token = self.task.workflow_args["token"]
        domain = self.task.workflow_args["domain"]
        template = self.task.workflow_args["template"]

        sandbox, created, step = get_or_create_sandbox(sandbox_id, url, token, domain, template)

        # oss_prefix = self.task.workflow_args["oss_prefix"]
        oss_config = self.task.workflow_args["oss"]
        dashscope_api_key = self.task.workflow_args["dashscope_api_key"]
        task_id = self.task.raw_task["task_id"]
        api_server_url = f"{self.model.api_address}/v1"
        dataset = run_workflow(sandbox, task_id, oss_config, dashscope_api_key, api_server_url)
        exps = []
        for data in dataset:
            exp = self.model.convert_messages_to_experience(data["messages"], data["tools"])
            exp.reward = float(data.get("judge_ok", 0.0))
            exps.append(exp)

        # sandbox.kill()  # 先不杀掉沙箱，方便调试和查看结果
        self.logger.info(
            f"Workflow finished. Sandbox {'created' if created else 'connected'} (ID: {sandbox.sandbox_id}). Collected {len(exps)} experiences."
        )

        return exps
