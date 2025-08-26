import openai
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import Task, SimpleWorkflow, WORKFLOWS
from trinity.common.rewards.reward_fn import RewardFn


@WORKFLOWS.register_module("simple_mm_workflow")
class SimpleMMWorkflow(SimpleWorkflow):
    """A workflow for simple single-round task."""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        self.reset(task)
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

    def reset(self, task: Task):
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reward_fn_args = task.reward_fn_args
        self.truth = task.raw_task[task.format_args.response_key] or task.truth

        reward_fn = task.reward_fn
        if isinstance(reward_fn, type) and issubclass(reward_fn, RewardFn):
            self.reward_fn: RewardFn = reward_fn(**self.reward_fn_args)
        else:
            raise ValueError("`reward_fn` must be a subclass of `RewardFn`")

        self.image_key = task.format_args.get("image_key", "image")
        self.video_key = task.format_args.get("video_key", "video")
        self.raw_mm_data = {}
        if task.raw_task.get(self.image_key) is not None:
            self.raw_mm_data["image"] = task.raw_task[self.image_key]
        if task.raw_task.get(self.video_key) is not None:
            self.raw_mm_data["video"] = task.raw_task[self.video_key]

    def run(self) -> List[Experience]:
        messages = self.format_messages()

        self.logger.debug("start chat")
        if self.raw_mm_data is not None:
            responses = self.model.chat_mm(messages, self.raw_mm_data, **self.rollout_args)
        else:
            responses = self.model.chat(messages, **self.rollout_args)
        for i, response in enumerate(responses):
            reward_dict = self.reward_fn(  # type: ignore [misc]
                response=response.response_text,  # type: ignore [arg-type]
                truth=self.truth,
            )

            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(reward_dict)
            reward = sum(reward_dict.values())
            response.reward = reward
            response.eid.run = i + self.run_id_base

            self.logger.debug(
                f"self.task_desc: {self.task_desc}, messages: {messages}, response: {response.response_text}, reward: {reward}"
            )
        return responses
