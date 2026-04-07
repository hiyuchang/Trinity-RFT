import asyncio
import os
from typing import Dict, List, Optional, Sequence, Union

import torch

from trinity.common.config import InferenceModelConfig
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel


class ExternalModel(InferenceModel):
    """Inference model backed by an external OpenAI-compatible API."""

    def __init__(self, config: InferenceModelConfig) -> None:
        super().__init__(config)
        self.model_version = 0
        self.client = None
        self.request_count = 0
        self.request_semaphore = asyncio.Semaphore(
            max(1, config.external_model_config.max_concurrent_requests)
        )
        self.api_base_url = os.getenv(config.external_model_config.base_url_env, "").rstrip("/")
        self.api_model_name = config.external_model_config.model_name or config.model_path
        if self.api_model_name is None:
            raise ValueError("`api_model_name` or `model_path` must be provided for openai_api.")

    async def prepare(self) -> None:
        if self.client is not None:
            return
        import openai

        self.client = openai.AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=os.getenv(self.config.external_model_config.api_key_env, ""),
            timeout=self.config.external_model_config.timeout,
        )
        self.logger.info(
            "Initialized openai_api engine with base_url=%s, model_name=%s, "
            "api_key_env=%s, max_concurrent_requests=%d",
            self.api_base_url,
            self.api_model_name,
            self.config.external_model_config.api_key_env,
            self.config.external_model_config.max_concurrent_requests,
        )

    def _build_experience(
        self,
        response_text: str,
        prompt_text: str = "",
        reward: Optional[float] = 0.0,
        metrics: Optional[dict[str, float]] = None,
        info: Optional[dict] = None,
    ) -> Experience:
        # Keep a minimal valid token tensor so existing pipelines can process single-turn data.
        tokens = torch.tensor([0, 0], dtype=torch.int32)
        logprobs = torch.tensor([0.0], dtype=torch.float32)
        return Experience(
            tokens=tokens,
            logprobs=logprobs,
            prompt_length=1,
            prompt_text=prompt_text,
            response_text=response_text,
            reward=reward,
            metrics=metrics or {},
            info=info or {},
        )

    async def _request_chat_completion(
        self, messages: List[Dict], **kwargs
    ) -> Sequence[Experience]:
        await self.prepare()
        assert self.client is not None
        self.request_count += 1
        request_id = self.request_count
        req_kwargs = {
            "model": self.api_model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_completion_tokens": kwargs.get(
                "max_completion_tokens", self.config.max_response_tokens
            ),
            "n": kwargs.get("n", 1),
        }
        self.logger.debug(
            "[openai_api][request=%d] model=%s max_completion_tokens=%s temperature=%s top_p=%s",
            request_id,
            req_kwargs["model"],
            req_kwargs["max_completion_tokens"],
            req_kwargs["temperature"],
        )
        async with self.request_semaphore:
            response = await self.client.chat.completions.create(**req_kwargs)

        usage_metrics = {}
        usage = getattr(response, "usage", None)
        if usage is not None:
            for usage_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_val = getattr(usage, usage_key, None)
                if isinstance(usage_val, (int, float)):
                    usage_metrics[f"usage/{usage_key}"] = float(usage_val)

        exps = []
        for choice in response.choices:
            content = choice.message.content or ""
            exps.append(
                self._build_experience(
                    response_text=content,
                    reward=0.0,
                    metrics=usage_metrics.copy(),
                    info={
                        "finish_reason": choice.finish_reason,
                        "choice_index": choice.index,
                    },
                )
            )
        return exps

    async def chat(self, messages: List[Dict], **kwargs) -> Sequence[Experience]:
        return await self.generate(messages, **kwargs)

    async def generate(self, prompt: Union[str, List[Dict]], **kwargs) -> Sequence[Experience]:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")
        return await self._request_chat_completion(messages, **kwargs)

    async def logprobs(self, token_ids: List[int], **kwargs) -> torch.Tensor:
        if not self.config.api_support_logprobs:
            raise NotImplementedError(
                "External API logprobs are disabled. Set `api_support_logprobs=true` "
                "and implement provider-specific logprob parsing if needed."
            )
        raise NotImplementedError(
            "logprobs for external OpenAI-compatible APIs is not implemented."
        )

    async def convert_messages_to_experience(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        del temperature
        if not messages:
            raise ValueError("`messages` must not be empty.")

        response_text = ""
        last = messages[-1]
        if last.get("role") == "assistant":
            content = last.get("content")
            response_text = content if isinstance(content, str) else ""

        exp = self._build_experience(response_text=response_text)
        exp.messages = messages
        exp.tools = tools
        return exp

    async def sync_model(self, model_version: int) -> int:
        self.model_version = model_version
        return model_version

    def get_model_version(self) -> int:
        return self.model_version

    def get_api_server_url(self) -> Optional[str]:
        # ModelWrapper appends `/v1` when building openai client base_url.
        if self.api_base_url.endswith("/v1"):
            return self.api_base_url[: -len("/v1")]
        return self.api_base_url
