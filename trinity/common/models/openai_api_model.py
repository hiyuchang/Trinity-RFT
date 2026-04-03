import asyncio
import os
from typing import Dict, List, Optional, Sequence, Union

import torch

from trinity.common.config import InferenceModelConfig
from trinity.common.experience import Experience
from trinity.common.models.model import InferenceModel


class OpenaiAPIModel(InferenceModel):
    """Inference model backed by an external OpenAI-compatible API."""

    def __init__(self, config: InferenceModelConfig) -> None:
        super().__init__(config)
        self.model_version = 0
        self.client = None
        self.request_count = 0
        self.request_semaphore = asyncio.Semaphore(max(1, config.api_max_concurrent_requests))
        self.api_base_url = os.getenv(config.api_base_url_env, "").rstrip("/")
        self.api_model_name = config.api_model_name or config.model_path
        if self.api_model_name is None:
            raise ValueError("`api_model_name` or `model_path` must be provided for openai_api.")

    async def prepare(self) -> None:
        if self.client is not None:
            return
        import openai

        self.client = openai.AsyncOpenAI(
            base_url=self.api_base_url,
            api_key=os.getenv(self.config.api_key_env, ""),
            timeout=self.config.api_timeout,
        )
        self.logger.info(
            "Initialized openai_api engine with base_url=%s, api_model_name=%s, "
            "api_key_env=%s, api_max_concurrent_requests=%d",
            self.api_base_url,
            self.api_model_name,
            self.config.api_key_env,
            self.config.api_max_concurrent_requests,
        )

    def _build_experience(self, response_text: str, prompt_text: str = "") -> Experience:
        # Keep a minimal valid token tensor so existing pipelines can process single-turn data.
        tokens = torch.tensor([0, 0], dtype=torch.int32)
        logprobs = torch.tensor([0.0], dtype=torch.float32)
        return Experience(
            tokens=tokens,
            logprobs=logprobs,
            prompt_length=1,
            prompt_text=prompt_text,
            response_text=response_text,
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
            "top_p": kwargs.get("top_p", self.config.top_p),
            "max_tokens": kwargs.get("max_tokens", self.config.max_response_tokens),
            "n": kwargs.get("n", 1),
        }
        self.logger.debug(
            "[openai_api][request=%d] model=%s max_tokens=%s temperature=%s top_p=%s",
            request_id,
            req_kwargs["model"],
            req_kwargs["max_tokens"],
            req_kwargs["temperature"],
            req_kwargs["top_p"],
        )
        async with self.request_semaphore:
            response = await self.client.chat.completions.create(**req_kwargs)

        exps = []
        for choice in response.choices:
            content = choice.message.content or ""
            exps.append(self._build_experience(response_text=content))
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

    async def convert_messages_to_experience(  # type: ignore[override]
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        temperature: Optional[float] = None,
    ) -> Experience:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        responses = await self.chat(messages, **kwargs)
        return responses[0]

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
