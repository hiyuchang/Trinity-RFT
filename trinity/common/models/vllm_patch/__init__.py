import asyncio
from logging import Logger

import vllm
from packaging.version import InvalidVersion
from packaging.version import parse as parse_version

from trinity.common.config import InferenceModelConfig

VLLM_VERSION_0120 = parse_version("0.12.0")
VLLM_VERSION_0170 = parse_version("0.17.0")
VLLM_VERSION_0191 = parse_version("0.19.1")


def vllm_patch():
    import transformers

    # Patch for Kimi-VL-A3B-Thinking
    if not hasattr(transformers.activations, "PytorchGELUTanh"):
        transformers.activations.PytorchGELUTanh = transformers.activations.GELUTanh

    trf_version = parse_version(transformers.__version__)
    vllm_version = parse_version(vllm.__version__)
    if trf_version >= parse_version("5.0.0"):
        if vllm_version < parse_version("0.16.0"):
            raise ImportError("Please upgrade vllm to 0.16.0 or above to use transformers>=5.0.0.")

        from transformers.configuration_utils import PreTrainedConfig

        original_init = PreTrainedConfig.__init__

        def new_init(self, *args, **kwargs):
            if "ignore_keys_at_rope_validation" in kwargs:
                kwargs["ignore_keys_at_rope_validation"] = set(
                    kwargs["ignore_keys_at_rope_validation"]
                )
            original_init(self, *args, **kwargs)

        PreTrainedConfig.__init__ = new_init


def get_vllm_version():
    try:
        vllm_version = parse_version(vllm.__version__)
    except InvalidVersion:
        # for self-compiled vllm,
        # we cannot parse the version, trait it as the lowest version we support
        vllm_version = parse_version("0.8.5")
    return vllm_version


def _get_api_server_runner(vllm_version):
    if vllm_version == VLLM_VERSION_0120:
        from trinity.common.models.vllm_patch.api_patch_v12 import (
            run_api_server_in_ray_actor_v12,
        )

        return run_api_server_in_ray_actor_v12

    if VLLM_VERSION_0120 < vllm_version < VLLM_VERSION_0170:
        from trinity.common.models.vllm_patch.api_patch_v13 import (
            run_api_server_in_ray_actor_v13,
        )

        return run_api_server_in_ray_actor_v13

    if VLLM_VERSION_0170 <= vllm_version <= VLLM_VERSION_0191:
        from trinity.common.models.vllm_patch.api_patch_v17 import (
            run_api_server_in_ray_actor_v17,
        )

        return run_api_server_in_ray_actor_v17

    raise ValueError(
        f"Unsupported vLLM version: {vllm.__version__}. "
        "This patch supports vLLM versions 0.12.0, (0.12.0, 0.17.0), and [0.17.0, 0.19.1]."
    )


def get_api_server(
    async_llm,
    host: str,
    port: int,
    config: InferenceModelConfig,
    logger: Logger,
):
    vllm_version = get_vllm_version()

    run_api_server_in_ray_actor = _get_api_server_runner(vllm_version)
    logger.info(f"Using vLLM API patch for version {vllm.__version__}")
    return asyncio.create_task(
        run_api_server_in_ray_actor(
            async_llm,
            host=host,
            port=port,
            model_path=config.model_path,  # type: ignore [arg-type]
            logger=logger,
            enable_auto_tool_choice=config.enable_auto_tool_choice,
            tool_call_parser=config.tool_call_parser,
            reasoning_parser=config.reasoning_parser,
            enable_log_requests=config.enable_log_requests,
            chat_template=config.chat_template,
        )
    )
