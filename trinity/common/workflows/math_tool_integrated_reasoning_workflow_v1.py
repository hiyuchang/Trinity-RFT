# -*- coding: utf-8 -*-
"""
Math Tool-Integrated Reasoning Workflow V1.

This workflow uses OpenAI's native tool calling feature for math problems with Python code execution.
The model can make tool calls to execute Python code, and provides a final answer after
multiple reasoning steps.

Key differences from V0:
- Prompt format strictly aligned with test_math_native_toolcall_collect_messages.py
- Uses exact_match for reward calculation instead of judge model
"""

import asyncio
import json
import os
import re
import sys
import tempfile
from typing import List, Optional

import openai
import shortuuid
import torch

from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, Task, Workflow


# Default system prompt is None to match test_math_native_toolcall_collect_messages.py
DEFAULT_SYSTEM_PROMPT = None


class PythonExecutionToolHandler:
    """Handler for Python code execution tool."""
    
    def __init__(self, timeout: float = 300):
        self.timeout = timeout
    
    def get_tool_definitions(self):
        """Get tool definitions in OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute Python code and return the output. Use this to perform calculations, verify solutions, or run any Python computations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "The Python code to execute. Remember to import necessary libraries like math, numpy, sympy before using them. Use print() to output results into the standard output.",
                            }
                        },
                        "required": ["code"],
                    },
                },
            }
        ]
    
    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute the specified tool."""
        if tool_name == "execute_python":
            return await self._execute_python(arguments["code"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _execute_python(self, code: str) -> str:
        """Execute Python code and return formatted output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, f"tmp_{shortuuid.uuid()}.py")
            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(code)

            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-u",
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                await asyncio.wait_for(proc.wait(), timeout=self.timeout)
                stdout, stderr = await proc.communicate()
                stdout_str = stdout.decode("utf-8")
                stderr_str = stderr.decode("utf-8")
                returncode = proc.returncode
            except asyncio.TimeoutError:
                returncode = -1
                try:
                    proc.terminate()
                    stdout, stderr = await proc.communicate()
                    stdout_str = stdout.decode("utf-8")
                    stderr_str = stderr.decode("utf-8") + f"\nTimeoutError: Code execution exceeded {self.timeout}s"
                except ProcessLookupError:
                    stdout_str = ""
                    stderr_str = f"TimeoutError: Code execution exceeded {self.timeout}s"

            # Format output as JSON for better parsing
            result = {
                "returncode": returncode,
                "stdout": stdout_str,
                "stderr": stderr_str,
            }
            return json.dumps(result, indent=2)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the answer from \\boxed{} notation.
    
    Handles nested braces correctly by counting brace depth.
    """
    if not text:
        return None
    
    # Find the last occurrence of \boxed
    idx = text.rfind("\\boxed")
    
    # Handle special case of \boxed (space instead of brace)
    if "\\boxed " in text:
        return text.split("\\boxed ")[-1].split("$")[0].strip()
    
    if idx < 0:
        # Also try \fbox as fallback
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    
    # Find the matching closing brace by counting brace depth
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    
    while i < len(text):
        if text[i] == "{":
            num_left_braces_open += 1
        if text[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx is None:
        return None
    
    # Extract the content between \boxed{ and the matching }
    boxed_string = text[idx:right_brace_idx + 1]
    
    # Remove the \boxed{ prefix and trailing }
    if "\\boxed " in boxed_string:
        left = "\\boxed "
        if boxed_string.startswith(left):
            return boxed_string[len(left):].strip()
    
    left = "\\boxed{"
    if boxed_string.startswith(left) and boxed_string.endswith("}"):
        return boxed_string[len(left):-1].strip()
    
    return None


def has_final_answer(text: str) -> bool:
    """Check if text contains a final answer in \\boxed{} format."""
    return extract_boxed_answer(text) is not None


def normalize_answer(answer: str) -> str:
    """Normalize answer for exact matching.
    
    This function removes common formatting differences to allow for more
    robust exact matching while still being stricter than LLM judge.
    """
    if not answer:
        return ""
    
    # Convert to string and strip whitespace
    answer = str(answer).strip()
    
    # Remove LaTeX formatting
    answer = answer.replace("\\text{", "").replace("}", "")
    answer = answer.replace("\\,", "").replace("\\!", "")
    answer = answer.replace("\\;", "").replace("\\:", "")
    answer = answer.replace("\\quad", "").replace("\\qquad", "")
    answer = answer.replace("$", "")
    
    # Remove common punctuation at the end
    answer = answer.rstrip(".,;:")
    
    # Normalize whitespace
    answer = " ".join(answer.split())
    
    # Convert to lowercase for case-insensitive comparison
    answer = answer.lower()
    
    return answer


def exact_match(candidate: str, ground_truth: str) -> bool:
    """Check if candidate exactly matches ground truth after normalization.
    
    Args:
        candidate: The candidate answer to evaluate
        ground_truth: The ground truth answer
        
    Returns:
        bool: True if exact match, False otherwise
    """
    normalized_candidate = normalize_answer(candidate)
    normalized_truth = normalize_answer(ground_truth)
    
    return normalized_candidate == normalized_truth


@WORKFLOWS.register_module("math_tool_integrated_reasoning_workflow_v1")
class MathToolIntegratedReasoningWorkflowV1(Workflow):
    """A workflow for math reasoning with integrated Python code execution using native tool calls.
    
    This workflow supports multi-turn interactions where the model can:
    1. Reason about the problem
    2. Use native OpenAI tool calling to execute Python code
    3. Iterate until a final answer is reached
    
    The final reward is calculated by exact matching the model's answer with the ground truth
    (no LLM judge used).
    
    Key differences from V0:
    - Prompt format strictly aligned with test_math_native_toolcall_collect_messages.py:
      Uses "\n Remember to return your final answer within \\boxed{}" instead of 
      " You should return your final answer within \\boxed{}"
    - Uses exact_match for reward calculation instead of LLM judge
    """

    is_async: bool = True
    can_reset: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        """Initialize the workflow.

        Args:
            task: Task object containing the problem and ground truth
            model: The main model wrapper for agent inference
            auxiliary_models: List of auxiliary models (not used in v1, kept for compatibility)
        """
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        
        # Get OpenAI async client from model (similar to bcp_simple_react_workflow)
        self.openai_async_client = model.get_openai_async_client()
        self.model_name = self.openai_async_client.model_path
        
        # Initialize tool handler
        self.tool_handler = None  # Will be initialized in reset
        
        self.reset(task)

    def reset(self, task: Task):
        """Reset the workflow with a new task."""
        self.workflow_args = task.workflow_args

        # Workflow configuration
        self.max_turns = int(self.workflow_args.get("max_turns", 10))
        self.code_timeout = float(self.workflow_args.get("code_timeout", 300))
        self.only_train_success = bool(self.workflow_args.get("only_train_success", False))

        # Task information
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth
        
        # Format args (system_prompt defaults to None like test_math_native_toolcall_collect_messages.py)
        self.format_args = task.format_args
        self.system_prompt = getattr(task.format_args, 'system_prompt', None) if task.format_args else None
        self.reply_prefix = getattr(task.format_args, 'reply_prefix', None) if task.format_args else None
        
        self.rollout_args = task.rollout_args
        
        # Initialize tool handler
        self.tool_handler = PythonExecutionToolHandler(timeout=self.code_timeout)

    async def _run_conversation_with_tools(
        self,
        messages: list,
        temperature: float = 0.7,
    ) -> tuple[str, str, int, int]:
        """Run conversation loop with function calling.
        
        This follows the workflow from test_math_native_toolcall_collect_messages.py.

        Args:
            messages: Initial messages
            temperature: Sampling temperature

        Returns:
            Tuple of (status, final_response_text, successful_tool_calls, total_tool_calls)
        """
        final_response_text = ""
        tools = self.tool_handler.get_tool_definitions()
        successful_tool_calls = 0
        total_tool_calls = 0
        
        try:
            for turn in range(self.max_turns):
                self.logger.debug(f"Turn {turn + 1}/{self.max_turns}")
                
                # Call the model with native tool calling
                try:
                    response = await self.openai_async_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        temperature=temperature,
                        max_tokens=self.rollout_args.max_tokens or 8192,
                    )
                except Exception as e:
                    self.logger.error(f"API call failed at turn {turn}: {e}")
                    return "error", final_response_text, successful_tool_calls, total_tool_calls
                
                message = response.choices[0].message
                response_text = message.content if message.content else ""
                final_response_text = response_text
                
                # Add assistant message to conversation
                messages.append(message)
                
                self.logger.debug(f"Turn {turn}: Response length: {len(response_text)}")
                
                # Check for final answer
                if response_text and has_final_answer(response_text):
                    self.logger.debug(f"Turn {turn}: Found final answer, ending conversation")
                    return "completed", final_response_text, successful_tool_calls, total_tool_calls
                
                # Handle tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        function_name = tool_call.function.name
                        total_tool_calls += 1
                        
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse tool arguments: {e}")
                            function_args = {}
                        
                        self.logger.debug(f"Turn {turn}: Executing tool {function_name}")
                        
                        # Execute the tool
                        try:
                            result = await self.tool_handler.execute_tool(function_name, function_args)
                            
                            # Check if execution was successful
                            try:
                                result_dict = json.loads(result)
                                if result_dict.get("returncode") == 0 and result_dict.get("stdout", "").strip():
                                    successful_tool_calls += 1
                            except:
                                pass
                            
                            # Add tool response to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": result,
                            })
                            
                        except Exception as e:
                            error_msg = f"Error executing {function_name}: {str(e)}"
                            self.logger.warning(error_msg)
                            
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_msg,
                            })
                    
                    # Continue to next iteration
                    continue
                
                # No tool calls and no answer - prompt the model
                if not response_text or (not has_final_answer(response_text) and not message.tool_calls):
                    self.logger.debug(f"Turn {turn}: No tool calls or final answer found")
                    messages.append({
                        "role": "user",
                        "content": "Please either use the execute_python tool to perform calculations, or provide your final answer using \\boxed{}."
                    })
                    continue
            
            # Hit max turns
            self.logger.warning(f"Conversation hit max turns ({self.max_turns})")
            return "incomplete", final_response_text, successful_tool_calls, total_tool_calls
            
        except Exception as e:
            # Catch all other exceptions (e.g., model context length exceeded, unexpected errors)
            self.logger.error(f"Unexpected error in conversation loop: {type(e).__name__}: {str(e)}")
            return "incomplete", final_response_text, successful_tool_calls, total_tool_calls

    async def run_async(self):
        """Run the agent asynchronously to generate experiences.

        Returns:
            List of Experience objects for RL training
        """
        # Prepare initial messages (strictly following test_math_native_toolcall_collect_messages.py)
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Format aligned with test_math_native_toolcall_collect_messages.py line 305:
        # problem + "\n Remember to return your final answer within \\boxed{}"
        messages.append({
            "role": "user",
            "content": self.task_desc + "\n Remember to return your final answer within \\boxed{}"
        })
        
        if self.reply_prefix:
            messages.append({"role": "assistant", "content": self.reply_prefix})
        
        # Run conversation with tools
        status, final_response_text, successful_tool_calls, total_tool_calls = await self._run_conversation_with_tools(
            messages,
            temperature=self.rollout_args.temperature,
        )
        
        self.logger.info(f"Conversation status: {status}")
        
        # Calculate reward using exact match (no judge model)
        reward = await self.calculate_reward_async(final_response_text)
        
        self.logger.info(f"Task result - Reward: {reward}")
        
        # Extract experiences from model history
        experiences = self.model.extract_experience_from_history(clear_history=True)
        
        self.logger.debug(f"Experiences extracted: {len(experiences)}")
        
        # Process experiences
        for i, experience in enumerate(experiences):
            experience.eid.step = i
            experience.reward = reward
            
            # Add agent-specific metrics
            agent_metrics = {
                "successful_tool_calls": successful_tool_calls,
                "total_tool_calls": total_tool_calls,
                "accuracy": reward,
                "is_completed": 1.0 if status == "completed" else 0.0,
            }
            
            if experience.metrics is None:
                experience.metrics = {}
            experience.metrics.update(agent_metrics)
        
        # Handle only_train_success mode: if status is not "completed", keep only one experience with reward=0 and action_mask=0
        if self.only_train_success and status != "completed" and experiences:
            self.logger.info(
                f"only_train_success=True and status={status} (not completed), "
                f"reducing {len(experiences)} experiences to 1 with reward=0 and action_mask=0"
            )
            # Keep only the first experience
            first_exp = experiences[0]
            
            # Set reward to 0
            first_exp.reward = 0.0
            
            # Set action_mask to all zeros if it exists
            if first_exp.action_mask is not None:
                first_exp.action_mask = torch.zeros_like(first_exp.action_mask)
            
            # Update metrics
            if first_exp.metrics is None:
                first_exp.metrics = {}
            first_exp.metrics.update({
                "successful_tool_calls": successful_tool_calls,
                "total_tool_calls": total_tool_calls,
                "accuracy": 0.0,
                "is_completed": 0.0,
                "filtered_by_only_train_success": True,
            })
            
            # Keep only the first experience
            experiences = [first_exp]
        
        if experiences:
            self.logger.info(
                f"Returning {len(experiences)} experiences, "
                f"run_id: {str(experiences[-1].eid.run)}, "
                f"final reward: {experiences[-1].reward}"
            )
        else:
            self.logger.warning("No valid experiences to return.")
        
        return experiences

    async def calculate_reward_async(self, response_text: str) -> float:
        """Calculate reward by exact matching the model's answer with ground truth.

        Args:
            response_text: The full response text from the model.

        Returns:
            The reward (1.0 for correct, 0.0 for incorrect).
        """
        # Extract the answer
        extracted_answer = extract_boxed_answer(response_text)
        
        if extracted_answer is None:
            self.logger.warning("No answer found in response, assigning reward 0.0")
            return 0.0
        
        # Use exact match to check correctness
        is_correct = exact_match(extracted_answer, str(self.truth))
        
        self.logger.debug(
            f"Extracted answer: {extracted_answer}, "
            f"Ground truth: {self.truth}, "
            f"Normalized candidate: {normalize_answer(extracted_answer)}, "
            f"Normalized truth: {normalize_answer(str(self.truth))}, "
            f"Correct: {is_correct}"
        )
        
        return 1.0 if is_correct else 0.0
