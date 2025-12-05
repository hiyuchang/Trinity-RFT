# -*- coding: utf-8 -*-
"""TIR Reward Function Class."""
import re

from trinity.common.rewards.reward_fn import REWARD_FUNCTIONS, RewardFn
from trinity.utils.log import get_logger

logger = get_logger(__name__)


@REWARD_FUNCTIONS.register_module("tir_reward")
class TIRRewardFn(RewardFn):
    """A reward function for Tool-Integrated Reasoning (TIR) tasks."""

    def __init__(self, question, truth, answer_type=None, auxiliary_models=None, **kwargs):
        self.question = question
        self.truth = truth
        self.answer_type = answer_type
        self.auxiliary_models = auxiliary_models
        self.kwargs = kwargs
        self.use_llm_judge = False
        if kwargs.get("all_llm_judge", False):
            self.use_llm_judge = True
        if kwargs.get("case_by_case", False):
            if self.answer_type is None:
                self.use_llm_judge = False
            else:
                self.use_llm_judge = self.answer_type == "text"
                logger.debug(f"answer_type: {self.answer_type}, truth: {self.truth}")

    async def __call__(self, response, **kwargs):
        if self.use_llm_judge:
            is_correct = await self.judge_equal(response)
            logger.debug(f"is_correct: {is_correct}")
            return {"reward": 1.0 if is_correct else 0.0}
        else:
            import importlib.util
            import sys

            file_path = "benchmark/plugins/guru_math/naive_dapo.py"

            # 创建模块规范
            spec = importlib.util.spec_from_file_location("naive_dapo", file_path)

            # 从规范创建模块
            naive_dapo_module = importlib.util.module_from_spec(spec)

            # 将模块添加到 sys.modules（可选，但推荐）
            sys.modules["naive_dapo"] = naive_dapo_module

            # 执行模块
            spec.loader.exec_module(naive_dapo_module)

            # 现在可以调用 compute_score 函数
            result = naive_dapo_module.compute_score(response, self.truth, {})
            logger.debug(f"result: {result}")
            return {"reward": result["acc"]}

    async def judge_equal(
        self,
        answer: str,
    ) -> bool:
        """Use LLM to judge whether answer equals truth.

        Adapted from https://github.com/open-compass/CompassVerifier
        """
        if not self.auxiliary_models:
            logger.error("Error: auxiliary_models is required for LLM judge")
            return False
        judger = self.auxiliary_models[0]
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
                    question=self.question, gold_answer=self.truth, llm_response=answer
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
        return final_judgment == "A"


def process_judgment(judgment_str: str) -> str:
    """Process the judgment string to extract the final judgment letter."""
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
