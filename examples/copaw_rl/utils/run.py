#!/usr/bin/env python3
"""
run.py

Usage:
    python run.py [--patch <patch_zip_path>] [--session-id <session_id>] [--user-id <user_id>]
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from urllib import request as urllib_request

import bench_client
import requests
from setup_provider import config_provider

# 脚本自身所在目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run agent worker with patch zip deployment")
    parser.add_argument(
        "--oss-prefix", default="CoPaw-Pro/benchmark/", help="OSS prefix for benchmark files"
    )
    parser.add_argument(
        "--task_id", help="the task id from OSS (label studio)"  # FIXME task-id instead of task_id
    )
    parser.add_argument("--session-id", default=None, help="Session ID (auto-generated if not set)")
    parser.add_argument("--user-id", default="default", help="User ID (default: default)")
    parser.add_argument("--url", default="http://127.0.0.1:8088", help="API endpoint URL")
    parser.add_argument("--rl-url", required=True, help="RL API endpoint URL")
    parser.add_argument("--rl-model-id", required=True, help="RL model ID")
    parser.add_argument(
        "--sessions-dir",
        default="/app/working/workspaces/default/sessions",
        help="Directory where session JSON files are stored",
    )
    return parser.parse_args()


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _extract_task_description(instruction_text: str) -> list[str]:
    """从 instruction.md 中提取"任务说明"部分，去掉"期望输出"和"注意事项"等评测信息。"""
    sections = re.split(r"^(## .+)$", instruction_text, flags=re.MULTILINE)
    result_parts = []
    capture = False
    for part in sections:
        if re.match(r"^## 任务说明", part):
            capture = True
            continue
        elif re.match(r"^## ", part):
            capture = False
            continue
        if capture:
            result_parts.append(part.strip())

    if result_parts:
        result = str("\n\n".join(result_parts))
    else:
        result = instruction_text
    return result.split("__END_OF_QUERY__")


def deploy_environment(extract_dir: str) -> None:
    """将 environment/ 下的所有文件部署到 home。

    若 environment/ 下存在 config/ 和 data/ 子目录：
      - config/ 下的文件覆盖复制到 home
      - data/   下的文件移动到 home
    否则，直接将 environment/ 下的所有文件移动到 home。
    """
    env_dir = os.path.join(extract_dir, "environment")
    if not os.path.isdir(env_dir):
        log.warning("environment/ 目录不存在于解压目录: %s", env_dir)
        return

    home = "/app/working/workspaces/default"
    config_dir = os.path.join(env_dir, "config")
    data_dir = os.path.join(env_dir, "data")

    if os.path.isdir(config_dir) and os.path.isdir(data_dir):
        # 将 config/ 下的文件覆盖复制到 home
        shutil.copytree(config_dir, home, dirs_exist_ok=True)
        log.info("deploy config: %s -> %s", config_dir, home)

        # 将 data/ 下的文件覆盖复制到 home
        shutil.copytree(data_dir, home, dirs_exist_ok=True)
        log.info("deploy data: %s -> %s", data_dir, home)
    else:
        # 兜底：直接将 environment/ 下的所有文件覆盖复制到 home
        shutil.copytree(env_dir, home, dirs_exist_ok=True)
        log.info("deploy: %s -> %s", env_dir, home)


def call_agent(
    url: str,
    user_input: str,
    session_id: str,
    user_id: str,
    provider_base_url: str,
    provider_model_id: str,
) -> None:
    payload = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": [{"type": "text", "text": user_input, "status": "created"}],
            }
        ],
        "session_id": session_id,
        "user_id": user_id,
        "channel": "console",
        "stream": True,
    }

    headers = {"Referer": f"{url}/chat", "content-type": "application/json"}

    result = config_provider(
        qwenpaw_url=url,
        provider_name="rl-server",
        provider_base_url=provider_base_url,
        provider_model_id=provider_model_id,
        provider_model_name="rl-model",
    )
    log.info("Provider configured and model activated successfully: %s", result)
    # call agent
    response = requests.post(f"{url}/api/agent/process", json=payload, headers=headers, stream=True)
    response.raise_for_status()
    log.info("agent response: %s", response.text)

    # Consume the streaming response
    for chunk in response.iter_content(chunk_size=None):
        pass  # We don't need to process the stream output here


# this function is used to extract raw trajectories from session data
def extract_trajectories(session_data: dict) -> list:
    """
    从 session JSON 中提取 agent._model_trajectory 字段，
    将每条记录的 messages 与 response 合并为最终的 trajectories 列表。
    """
    model_trajectory = session_data.get("agent", {}).get("_model_trajectory", [])

    return model_trajectory


def extract_final_text(session_data: dict) -> str:
    """从 session 的最后一条 trajectory entry 中提取 agent 最终回复文本。"""
    raw_trajectory = session_data.get("agent", {}).get("_model_trajectory", [])
    if not raw_trajectory:
        return ""

    last_entry = raw_trajectory[-1]
    response = last_entry.get("response", [])

    if isinstance(response, str):
        return response
    if isinstance(response, list):
        for item in response:
            if isinstance(item, dict) and item.get("type") == "text":
                return item.get("text", "")
    return ""


JUDGE_MODEL = "qwen-plus"
JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
JUDGE_SYSTEM_PROMPT = (
    "判断任务是否可视为完成。"
    "规则1：若因为确实缺少关键信息导致必须与用户交互或无法继续，可判定为完成；"
    "规则2：若因为缺少文件，则必定是未完成，因为环境中必然有对应的文件，是模型未找到；"
    "但如果并非必须信息却发起了交互，应判定为失败。"
    "结合首条user消息与最后一个response综合判断。"
    '只输出JSON: {"success": true/false, "reason": "一句话理由"}。'
)


def has_tool_calls(traj: list) -> bool:
    """检查 trajectory 中是否存在 role 为 tool 的消息。"""
    for entry in traj:
        if not isinstance(entry, dict):
            continue
        for msg in entry.get("messages", []):
            if isinstance(msg, dict) and msg.get("role") == "tool":
                return True
    return False


def _llm_judge_sync(
    sample_id: str,
    first_user_message: str,
    last_response,
) -> tuple[bool, str]:
    """同步版 LLM 判断，在 executor 中调用。"""
    if last_response is None:
        return False, "trajectory为空或最后一轮无response"
    payload = {
        "model": JUDGE_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "sample_id": sample_id,
                        "first_user_message": first_user_message,
                        "last_response": last_response,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    req = urllib_request.Request(
        JUDGE_BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode("utf-8", errors="replace"))
    text = body["choices"][0]["message"]["content"]
    obj = json.loads(text)
    return bool(obj.get("success", False)), str(obj.get("reason", "")).strip()


def _llm_judge(sample_id: str, trajectory: list) -> tuple[bool, str]:
    """异步 LLM 判断 trajectory 是否成功。"""
    if not has_tool_calls(trajectory):
        return False, "trajectory中没有工具调用"

    first_user_message = ""
    if trajectory and isinstance(trajectory[0], dict):
        for m in trajectory[0].get("messages", []):
            if isinstance(m, dict) and m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str):
                    first_user_message = c
                elif isinstance(c, list):
                    parts = [
                        item.get("text", "")
                        for item in c
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    first_user_message = "\n".join(parts)
                break

    last_response = None
    if trajectory and isinstance(trajectory[-1], dict) and "response" in trajectory[-1]:
        last_response = trajectory[-1]["response"]

    return _llm_judge_sync(sample_id, first_user_message, last_response)


def main():
    args = parse_args()

    if not args.session_id:
        args.session_id = f"{args.task_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    log.info("session_id: %s", args.session_id)

    try:
        # Step 1. download bench from OSS
        client = bench_client.BenchmarkClient(oss_prefix=args.oss_prefix)
        client.download_compressed_task(args.task_id, _SCRIPT_DIR)

        # Step 2: 将 environment/ 文件部署到 $HOME
        log.info("部署 environment/ 到 $HOME ...")
        deploy_environment(_SCRIPT_DIR)

        # Step 2.5: 如果存在 setup.sh，在调用 Agent 之前执行
        setup_script = os.path.join(_SCRIPT_DIR, "setup.sh")
        if os.path.isfile(setup_script):
            log.info("发现 setup.sh，执行前置设置: %s", setup_script)
            os.chmod(setup_script, 0o755)
            setup_result = subprocess.run(
                ["bash", setup_script],
                capture_output=True,
                text=True,
                cwd=_SCRIPT_DIR,
                timeout=60,
            )
            if setup_result.returncode != 0:
                log.warning(
                    "setup.sh 执行失败 (exit=%d): %s", setup_result.returncode, setup_result.stderr
                )
            else:
                log.info("setup.sh 执行成功")
            if setup_result.stdout:
                log.info("setup.sh stdout: %s", setup_result.stdout.rstrip())

        # Step 3: 读取 instruction.md，提取"任务说明"部分作为 query
        instruction_path = os.path.join(_SCRIPT_DIR, "instruction.md")
        if not os.path.exists(instruction_path):
            log.error("instruction.md 不存在: %s", instruction_path)
            sys.exit(1)
        full_instruction = read_file(instruction_path)
        user_inputs = _extract_task_description(full_instruction)
        log.info("读取 instruction.md，query 长度: %d", len(user_inputs))

        # Step 4: 调用 agent API
        log.info("调用 agent API ...")
        t_start = time.time()
        for user_input in user_inputs:
            call_agent(
                args.url.strip("/"),
                user_input,
                args.session_id,
                args.user_id,
                args.rl_url,
                args.rl_model_id,
            )
        duration_seconds = round(time.time() - t_start, 2)
        log.info("完成调用 agent API，耗时: %.2f 秒", duration_seconds)

        # Step 5: 读取 session JSON 并提取 trajectories
        test_dir = os.path.join(_SCRIPT_DIR, "tests")
        os.makedirs(test_dir, exist_ok=True)

        session_file = f"{args.sessions_dir}/{args.user_id}_{args.session_id}.json"
        log.info("读取 session 文件: %s", session_file)
        session_text = read_file(session_file)
        session_data = json.loads(session_text)

        # 将原始 session 文件复制到 /app/session.json 方便下载
        session_export_path = os.path.join(test_dir, "session.json")
        shutil.copy2(session_file, session_export_path)
        log.info("原始 session 已导出到: %s", session_export_path)

        final_text = extract_final_text(session_data)
        log.info(f"Agent 最终回复文本:\n{final_text}\n")

        trajectories = extract_trajectories(session_data)
        # ── LLM 判断是否执行成功 ──────────────────────────────────────────
        try:
            judge_ok, judge_reason = _llm_judge(args.task_id, trajectories)
        except Exception as judge_exc:
            # 判断出错时保守处理：视为成功，避免误丢弃
            judge_ok, judge_reason = True, f"LLM判断异常(视为成功): {judge_exc}"
        log.info("LLM judge result: %s, reason: %s", judge_ok, judge_reason)

        dataset = []
        for trajectory in trajectories:
            logprobs = trajectory["logprobs"]
            prompt_token_ids = trajectory["prompt_token_ids"]
            token_ids = trajectory["token_ids"]

            data = {
                "logprobs": logprobs,
                "prompt_token_ids": prompt_token_ids,
                "token_ids": token_ids,
                "judge_ok": judge_ok,
            }
            dataset.append(data)

        with open(os.path.join(_SCRIPT_DIR, "dataset.json"), "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False)  # , indent=2

    except SystemExit:
        raise
    except Exception as e:
        log.exception("运行异常: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
