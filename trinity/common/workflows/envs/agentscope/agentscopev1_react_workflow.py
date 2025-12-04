# -*- coding: utf-8 -*-
"""We include the customized math workflows in this file."""

import re
from typing import List, Optional

import openai

from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


@WORKFLOWS.register_module("agentscope_react_math_workflow")
class AgentScopeReactMathWorkflow(Workflow):
    """
    This workflow serves as an example of how to use the agentscope framework within the trinity workflow.
    We use the AgentScope V1 version here.
    """

    can_reset: bool = True
    is_async: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

        # make sure that we have the correct import
        try:
            from agentscope.formatter import OpenAIChatFormatter
            from agentscope.model import OpenAIChatModel
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)

        # get openai client from model
        self.openai_async_client = model.get_openai_async_client()
        self.model_name = self.openai_async_client.model_path

        self.agent_model = OpenAIChatModel(
            api_key="EMPTY",
            model_name=self.model_name,
            stream=False,
            generate_kwargs={
                "temperature": self.task.rollout_args.temperature,
                "max_tokens": self.task.rollout_args.max_tokens or 4096,
            },
        )
        self.agent_model.client = self.openai_async_client
        self.agent_model_formatter = OpenAIChatFormatter()
        self.reset(task)

    def reset(self, task: Task):
        self.system_prompt = """
You are an agent specialized in solving math problems with tools. Please solve the math problem given to you. You can write and execute Python code to perform calculation or verify your answer. You should return your final answer within \\boxed{{}}.
"""
        try:
            from agentscope.agent import ReActAgent
            from agentscope.memory import InMemoryMemory
            from agentscope.tool import Toolkit, execute_python_code
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)
        self.toolkit = Toolkit()
        self.toolkit.register_tool_function(execute_python_code)
        self.agent = ReActAgent(
            name="math_react_agent",
            sys_prompt=self.system_prompt,
            model=self.agent_model,
            formatter=self.agent_model_formatter,
            toolkit=self.toolkit,
            memory=InMemoryMemory(),
        )
        self.agent.set_console_output_enabled(False)
        # we set the openai client to the agent's model
        self.agent.model.client = self.openai_async_client

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        # we get the answer from gsm8k dataset
        try:
            if isinstance(self.truth, str) and "####" in self.truth:
                # GSM8K dataset
                self.answer = self.truth.split("####")[1].strip()
            else:
                self.answer = str(self.truth)
        except Exception as e:
            self.logger.debug(f"Error in getting answer from truth: {str(e)}")
            self.answer = str(self.truth)

        # we use the boxed format to evaluate the answer

    async def run_async(self):
        # make sure that we have the correct import
        try:
            from agentscope.message import Msg
            from pydantic import BaseModel, Field
        except ImportError as e:
            error_message = f"AgentScope is not installed. Please install the agentscope framework first before running the workflow. Error: {str(e)}"
            self.logger.error(error_message)
            raise ImportError(error_message)

        # provide the task to the react agent
        msg = Msg("user", self.task_desc, role="user")

        # Note that the main workflow can have arbitrary steps and include different logic
        class FinalResult(BaseModel):
            result: str = Field(
                description="Your solution of the given math problem. Put your final answer in boxed format, e.g., \\boxed{42}"
            )

        def extract_final_answer(result) -> str:
            """Extract the final answer from the agent's response."""
            try:
                if (
                    hasattr(result, "metadata")
                    and isinstance(result.metadata, dict)
                    and "result" in result.metadata
                ):
                    return result.metadata["result"]
                if hasattr(result, "content"):
                    if isinstance(result.content, dict) and "result" in result.content:
                        return result.content["result"]
                    return str(result.content)
                return str(result)
            except Exception as e:
                self.logger.warning(f"Extract final answer error: {e}. Raw: {result}")
                return str(result)

        result = await self.agent.reply(msg, structured_model=FinalResult)

        final_answer = extract_final_answer(result)

        is_correct = await self.verify_answer(final_answer, self.answer, **self.task.reward_fn_args)
        reward = 1.0 if is_correct else 0.0
        self.logger.debug(f"Reward: {reward}")
        experiences = self.model.extract_experience_from_history(clear_history=True)
        self.logger.debug(f"Experiences extracted len: {len(experiences)}")
        for i, experience in enumerate(experiences):
            experience.eid.step = i
            experience.reward = reward
            agent_metrics = {"react_memory_length": len(self.agent.memory.content)}
            if experience.metrics is None:
                experience.metrics = {}
            experience.metrics.update(agent_metrics)
        self.logger.debug(
            f"return experience len: {len(experiences)}, run_id: {str(experiences[-1].eid.run)}, final step reward: {experiences[-1].reward}"
        )
        return experiences

    async def verify_answer(self, answer, truth, **kwargs):
        use_llm_judge = False
        if kwargs.get("all_llm_judge", False):
            use_llm_judge = True
        if kwargs.get("case_by_case", False):
            answer_type = self.raw_task.get("answer_type", None)
            if answer_type is None:
                use_llm_judge = False
            else:
                use_llm_judge = answer_type == "text"
                print("answer_type: ", answer_type)

        if use_llm_judge:
            is_correct = await self.judge_equal(answer, truth)
        else:
            from benchmark.plugins.guru_math.naive_dapo import compute_score

            is_correct = compute_score(answer, truth, {})["acc"]
        return is_correct

    async def judge_equal(self, answer, truth):
        """Adated from https://github.com/open-compass/CompassVerifier"""
        judger = self.auxiliary_models[0] if self.auxiliary_models else None
        user_prompt = """
Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.
Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.
2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.
3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct.
4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct as long as it matches the standard answer, regardless of whether the reasoning process is correct. For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must be answered correctly and match the standard answer exactly to be deemed correct.
5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the candidate's answer is consistent with the standard answer.
6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: INVALID
Just return the letters "A", "B", or "C", with no text around it.
Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>
Judging the correctness of the candidate's answer:
"""
        messages = [
            {
                "role": "user",
                "content": user_prompt.format(
                    question=self.task_desc, gold_answer=truth, llm_response=answer
                ),
            },
        ]
        completion = await judger.chat.completions.create(
            model=judger.model_path,
            messages=messages,
            stream=False,
            temperature=0.0,
        )
        response = completion.choices[0].message.content
        final_judgment = process_judgment(response)
        return True if final_judgment == "A" else False


def process_judgment(judgment_str: str) -> str:
    # First try to find the exact \boxed{letter} pattern
    boxed_matches = re.findall(r"boxed{([A-C])}", judgment_str)
    if boxed_matches:
        return boxed_matches[-1]

    # Directly return the judgment if it is A, B, or C
    if judgment_str in ["A", "B", "C"]:
        return judgment_str
    else:
        final_judgment_str = judgment_str.split("Final Judgment:")[-1]
        matches = re.findall(r"\(([A-C])\)*", final_judgment_str)
        if matches:
            return matches[-1]
        matches = re.findall(r"([A-C])", final_judgment_str)
        if matches:
            return matches[-1]
        return ""
