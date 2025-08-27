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
        self.system_prompt = """You are a helpful assistant that solves MATH problems. You should first thinks about the reasoning process in mind and then provides the user with the answer. You should present your reasoning process using the format: <think>\n ...your reasoning process here... </think>\n first. You should always include your final answer in \\boxed{} as closed-form results."""  # TODO: check
        self.reply_prefix = task.format_args.reply_prefix
        self.reward_fn_args = task.reward_fn_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.raw_task[task.format_args.response_key] or task.truth

        # TODO
        self.reward_fn = self.compute_reward

        self.image_key = task.format_args.image_key
        self.video_key = task.format_args.video_key
        self.raw_mm_data = {}
        if self.image_key and task.raw_task.get(self.image_key) is not None:
            self.raw_mm_data["image"] = task.raw_task[self.image_key]
        if self.video_key and task.raw_task.get(self.video_key) is not None:
            self.raw_mm_data["video"] = task.raw_task[self.video_key]

    def run(self) -> List[Experience]:
        messages = self.format_messages()

        # TODO: test generate_mm
        self.logger.debug("start chat")
        # self.logger.debug(f"run messages: {messages}")
        # self.logger.debug(f"run self.raw_mm_data: {self.raw_mm_data}")
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

            # self.logger.debug(
            #     f"self.task_desc: {self.task_desc},  response: {response.response_text}, reward: {reward}"
            # )
        return responses

    def compute_reward(self, response, truth) -> float:
        from mathruler.grader import extract_boxed_content, grade_answer

        answer = extract_boxed_content(response)
        if grade_answer(answer, truth):
            return {"accuracy": 1.0}  # correct answer
        return {"accuracy": 0.0}  # wrong answer
