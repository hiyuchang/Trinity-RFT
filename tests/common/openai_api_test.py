import asyncio
import gc
import os
import unittest

import ray

from tests.tools import get_model_path, get_template_config
from trinity.common.config import ExternalModelConfig, InferenceModelConfig
from trinity.common.models import create_explorer_models
from trinity.common.models.model import ModelWrapper
from trinity.common.models.openai_api_model import OpenaiAPIModel


async def prepare_engines(engines, auxiliary_engines):
    prepare_refs = []
    for engine in engines:
        prepare_refs.append(engine.prepare.remote())
    for models in auxiliary_engines:
        for engine in models:
            prepare_refs.append(engine.prepare.remote())
    await asyncio.gather(*prepare_refs)


class TestOpenaiAPIModel(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        ray.init(ignore_reinit_error=True, namespace="trinity_unittest")
        gc.collect()

    @classmethod
    def tearDownClass(cls):
        ray.shutdown(_exiting_interpreter=True)

    async def asyncSetUp(self):
        model_path = get_model_path()
        # Part 1: bootstrap a local OpenAI-compatible endpoint via vLLM.
        config = get_template_config()
        config.mode = "explore"
        config.model.model_path = model_path
        config.explorer.rollout_model.engine_type = "vllm"
        config.explorer.rollout_model.engine_num = 1
        config.explorer.rollout_model.tensor_parallel_size = 1
        config.explorer.rollout_model.enable_openai_api = True
        config.check_and_update()

        self.engines, self.auxiliary_engines = create_explorer_models(config)
        self.vllm_wrapper = ModelWrapper(self.engines[0], enable_history=False)
        await prepare_engines(self.engines, self.auxiliary_engines)
        await self.vllm_wrapper.prepare()

        openai_client = self.vllm_wrapper.get_openai_client()
        self.model_name = openai_client.models.list().data[0].id

        self.base_url_env = "TRINITY_OPENAI_BASE_URL_TEST"
        self.api_key_env = "TRINITY_OPENAI_API_KEY_TEST"
        os.environ[self.base_url_env] = f"{self.vllm_wrapper.api_address}/v1"
        os.environ[self.api_key_env] = "EMPTY"
        self.model_path = model_path
        print(
            f"Model is prepared at {self.vllm_wrapper.api_address}/v1, model_name: {self.model_name}"
        )

    async def test_openai_api_model_basic(self):
        # Part 2: verify OpenaiAPIModel can call the endpoint correctly.
        model = OpenaiAPIModel(
            InferenceModelConfig(
                engine_type="external",
                model_path=self.model_path,
                max_prompt_tokens=1024,
                enable_prompt_truncation=True,
                external_model_config=ExternalModelConfig(
                    base_url_env=self.base_url_env,
                    api_key_env=self.api_key_env,
                    model_name=self.model_name,
                    max_concurrent_requests=4,
                ),
            )
        )

        generate_exps = await model.generate("Say hello in one sentence.", n=1, max_tokens=16)
        self.assertEqual(len(generate_exps), 1)
        self.assertTrue(len(generate_exps[0].response_text) > 0)

        messages = [
            {"role": "system", "content": "You are an assistant. Answer the question briefly."},
            {"role": "user", "content": [{"type": "text", "text": "What is 1+1?"}]},
        ]
        chat_exps = await model.chat(messages, n=1, max_tokens=32)
        self.assertEqual(len(chat_exps), 1)
        self.assertTrue(len(chat_exps[0].response_text) > 0)
