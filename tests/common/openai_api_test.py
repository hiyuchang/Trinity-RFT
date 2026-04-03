import asyncio
import gc
import os
import unittest
from pathlib import Path

import ray

from trinity.common.config import InferenceModelConfig, load_config
from trinity.common.constants import MODEL_PATH_ENV_VAR
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
        model_path = os.environ.get(MODEL_PATH_ENV_VAR)
        if not model_path:
            raise unittest.SkipTest(
                f"Please set `export {MODEL_PATH_ENV_VAR}=<your_model_dir>` before running this test."
            )

        # Part 1: bootstrap a local OpenAI-compatible endpoint via vLLM.
        config_path = Path(__file__).resolve().parents[1] / "template" / "config.yaml"
        config = load_config(str(config_path))
        config.mode = "explore"
        config.ray_namespace = "trinity_unittest"
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

    async def test_openai_api_model_basic(self):
        # Part 2: verify OpenaiAPIModel can call the endpoint correctly.
        model = OpenaiAPIModel(
            InferenceModelConfig(
                engine_type="openai_api",
                model_path=self.model_path,
                api_base_url_env=self.base_url_env,
                api_key_env=self.api_key_env,
                api_model_name=self.model_name,
                api_max_concurrent_requests=2,
                max_prompt_tokens=8,
                enable_prompt_truncation=True,
            )
        )

        generate_exps = await model.generate("Say hello in one sentence.", n=1, max_tokens=16)
        self.assertEqual(len(generate_exps), 1)
        self.assertTrue(len(generate_exps[0].response_text) > 0)

        messages = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": [{"type": "text", "text": "What is 1+1?"}]},
        ]
        chat_exps = await model.chat(messages, n=1, max_tokens=16)
        self.assertEqual(len(chat_exps), 1)
        self.assertTrue(len(chat_exps[0].response_text) > 0)

        long_prompt_exps = await model.generate("hello " * 1024, n=2)
        self.assertEqual(len(long_prompt_exps), 2)
        self.assertTrue(all(exp.truncate_status == "prompt_truncated" for exp in long_prompt_exps))

        token_len = await model.get_message_token_len(messages)
        self.assertGreater(token_len, 0)
