import argparse
import json

import requests


def _detect_api_base(qwenpaw_url: str, timeout: int = 10) -> str:
    """Detect whether the service is mounted under /api.

    QwenPaw in this repo exposes most endpoints under /api, but some
    deployments may already include /api in user-provided base URL.
    """
    root = qwenpaw_url.rstrip("/")
    if root.endswith("/api"):
        return root

    api_candidate = f"{root}/api"
    check_urls = [f"{api_candidate}/version", f"{root}/api/version"]
    for url in check_urls:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.ok:
                return api_candidate
        except requests.RequestException:
            continue

    # Fallback to /api because current QwenPaw app mounts API routers there.
    return api_candidate


def _safe_json(resp: requests.Response):
    try:
        return resp.json()
    except ValueError:
        return {"text": resp.text}


def _add_model_if_needed(base: str, provider_id: str, model_id: str) -> None:
    """Best-effort model registration for provider before activation."""
    resp = requests.post(
        f"{base}/models/{provider_id}/models",
        json={"id": model_id, "name": model_id},
    )
    # 201: added; 400: maybe already exists; ignore here.
    _ = resp


def _activate_model(
    base: str,
    provider_id: str,
    provider_model_id: str,
    agent_id: str = None,
):
    payload = {
        "provider_id": provider_id,
        "model": provider_model_id,
        "scope": "global",
    }
    if agent_id:
        payload = {
            "provider_id": provider_id,
            "model": provider_model_id,
            "scope": "agent",
            "agent_id": agent_id,
        }

    activate_resp = requests.put(
        f"{base}/models/active",
        json=payload,
    )
    activate_resp.raise_for_status()
    return activate_resp.json()


def _probe_multimodal(
    base: str,
    provider_id: str,
    provider_model_id: str,
):
    """Probe multimodal capability after activation to refresh metadata."""
    probe_resp = requests.post(
        f"{base}/models/{provider_id}/models/{provider_model_id}/probe-multimodal"
    )
    probe_resp.raise_for_status()
    return probe_resp.json()


def config_provider(
    qwenpaw_url: str,
    provider_name: str,
    provider_base_url: str,
    provider_model_id: str,
    provider_model_name: str = None,
    provider_api_key: str = "",
    provider_chat_model: str = "OpenAIChatModel",
    agent_id: str = None,
):
    base = _detect_api_base(qwenpaw_url)
    provider_id = provider_name

    # 1. 创建自定义 provider
    create_resp = requests.post(
        f"{base}/models/custom-providers",
        json={
            "id": provider_id,
            "name": provider_name,
            "default_base_url": provider_base_url,
            "api_key_prefix": "",
            "chat_model": provider_chat_model,
            "models": [
                {
                    "id": provider_model_id,
                    "name": provider_model_name or provider_model_id,
                }
            ],
        },
    )

    # 如果 provider 已存在（常见为 400），尝试更新配置
    if create_resp.status_code not in (200, 201):
        config_resp = requests.put(
            f"{base}/models/{provider_id}/config",
            json={
                "api_key": provider_api_key or None,
                "base_url": provider_base_url or None,
                "chat_model": provider_chat_model,
            },
        )
        if config_resp.status_code not in (200, 201):
            raise requests.HTTPError(
                (
                    "Failed to create/update provider. "
                    f"create_status={create_resp.status_code}, "
                    f"create_detail={_safe_json(create_resp)}, "
                    f"config_status={config_resp.status_code}, "
                    f"config_detail={_safe_json(config_resp)}"
                ),
                response=config_resp,
            )
    else:
        # 更新 api_key（创建接口不含 api_key，需单独配置）
        if provider_api_key:
            config_resp = requests.put(
                f"{base}/models/{provider_id}/config",
                json={"api_key": provider_api_key},
            )
            config_resp.raise_for_status()

    # 2. 激活模型
    # 当前版本会校验模型必须先存在于 provider 中，先补充注册
    _add_model_if_needed(base, provider_id, provider_model_id)
    active = _activate_model(
        base=base,
        provider_id=provider_id,
        provider_model_id=provider_model_id,
        agent_id=agent_id,
    )
    probe = _probe_multimodal(base, provider_id, provider_model_id)
    return {"active": active, "probe": probe}


def config_builtin_provider(
    qwenpaw_url: str,
    provider_name: str,
    provider_model_id: str,
    provider_api_key: str = "",
    provider_base_url: str = "",
    provider_chat_model: str = "OpenAIChatModel",
    generate_kwargs: dict = None,
    agent_id: str = None,
):
    """Configure a built-in provider, add model, and activate it."""
    base = _detect_api_base(qwenpaw_url)
    provider_id = provider_name

    config_payload = {
        "api_key": provider_api_key or None,
        "base_url": provider_base_url or None,
        "chat_model": provider_chat_model,
        "generate_kwargs": generate_kwargs or {},
    }
    config_resp = requests.put(
        f"{base}/models/{provider_id}/config",
        json=config_payload,
    )
    config_resp.raise_for_status()

    _add_model_if_needed(base, provider_id, provider_model_id)
    active = _activate_model(
        base=base,
        provider_id=provider_id,
        provider_model_id=provider_model_id,
        agent_id=agent_id,
    )
    probe = _probe_multimodal(base, provider_id, provider_model_id)
    return {"active": active, "probe": probe}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configure and activate model providers in QwenPaw."
    )
    parser.add_argument(
        "--qwenpaw_url", type=str, required=True, help="Base URL of the QwenPaw service."
    )
    parser.add_argument(
        "--provider_type",
        type=str,
        default="custom",
        choices=["custom", "builtin"],
        help="Provider type: custom or builtin.",
    )
    parser.add_argument(
        "--provider_name",
        type=str,
        required=True,
        help="Provider ID/name, e.g. rl-server or dashscope.",
    )
    parser.add_argument(
        "--provider_base_url", type=str, default="", help="Base URL for provider API."
    )
    parser.add_argument(
        "--provider_model_id",
        type=str,
        required=True,
        help="Model ID to activate for the provider.",
    )
    parser.add_argument(
        "--provider_model_name",
        type=str,
        default=None,
        help="Model display name for custom provider (optional).",
    )
    parser.add_argument(
        "--provider_api_key",
        type=str,
        default="",
        help="API key for the custom provider (if required).",
    )
    parser.add_argument(
        "--provider_chat_model",
        type=str,
        default="OpenAIChatModel",
        help="Chat model type for the provider.",
    )
    parser.add_argument(
        "--generate_kwargs_json",
        type=str,
        default="{}",
        help="JSON string for generate_kwargs, e.g. '{\"temperature\":0.2}'.",
    )
    parser.add_argument(
        "--agent_id",
        type=str,
        default=None,
        help="Agent ID to scope the model activation (optional).",
    )

    args = parser.parse_args()
    try:
        generate_kwargs = json.loads(args.generate_kwargs_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--generate_kwargs_json must be valid JSON") from exc

    if args.provider_type == "builtin":
        result = config_builtin_provider(
            qwenpaw_url=args.qwenpaw_url,
            provider_name=args.provider_name,
            provider_model_id=args.provider_model_id,
            provider_api_key=args.provider_api_key,
            provider_base_url=args.provider_base_url,
            provider_chat_model=args.provider_chat_model,
            generate_kwargs=generate_kwargs,
            agent_id=args.agent_id,
        )
    else:
        result = config_provider(
            qwenpaw_url=args.qwenpaw_url,
            provider_name=args.provider_name,
            provider_base_url=args.provider_base_url,
            provider_model_id=args.provider_model_id,
            provider_model_name=args.provider_model_name,
            provider_api_key=args.provider_api_key,
            provider_chat_model=args.provider_chat_model,
            agent_id=args.agent_id,
        )
    print("Provider configured and model activated successfully:", result)


# python setup_provider.py --qwenpaw_url http://127.0.0.1:8088 --provider_name rl-server --provider_base_url http://10.56.28.162:25015 --provider_model_id /mnt/data/chenyushuo.cys/rlhf_space/models/Qwen3.5-4B
# python setup_provider.py --qwenpaw_url http://127.0.0.1:8088 --provider_type builtin --provider_name dashscope --provider_base_url https://dashscope.aliyuncs.com/compatible-mode/v1 --provider_model_id qwen3.5-flash --provider_api_key xxxx
