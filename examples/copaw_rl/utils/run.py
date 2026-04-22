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

import bench_client
import requests
import yaml
from export_training_data import export_training_data
from setup_provider import config_provider

# 脚本自身所在目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(_SCRIPT_DIR, "summary.json")

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
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Evaluation mode: run grader and output summary.json",
    )
    return parser.parse_args()


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# 公共评测模块注入
# ---------------------------------------------------------------------------

_CONFTEST_TEMPLATE = '''\
"""自动生成的 conftest.py — 提供公共 pytest fixtures。"""
import json
import os
import pytest

SESSION_FILE = os.environ.get("SESSION_FILE", "")

@pytest.fixture
def session():
    """公共 fixture：加载并校验 session JSON。"""
    if not SESSION_FILE or not os.path.isfile(SESSION_FILE):
        pytest.fail("Session 文件不存在")
    with open(SESSION_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            pytest.fail("Session 文件不是有效 JSON")
    if not data or "agent" not in data:
        pytest.fail("Session 缺少 agent 字段")
    return data
'''


def _inject_shared_eval_module(test_dir: str) -> None:
    """将公共评测模块 copaw_eval.py 和 conftest.py 写入 tests/ 目录。

    copaw_eval.py 的源码由 batch_run.py 在注入时嵌入到全局变量
    COPAW_EVAL_SOURCE 中。如果该变量不存在（本地调试），则尝试从
    injection_server/ 同级目录读取。
    """
    os.makedirs(test_dir, exist_ok=True)

    # # 1. 写入 copaw_eval.py
    # source = globals().get("COPAW_EVAL_SOURCE", "")
    # if not source:
    #     local_path = os.path.join(_SCRIPT_DIR, "copaw_eval.py")
    #     if os.path.isfile(local_path):
    #         with open(local_path, "r", encoding="utf-8") as f:
    #             source = f.read()
    # if source:
    #     dest = os.path.join(test_dir, "copaw_eval.py")
    #     with open(dest, "w", encoding="utf-8") as f:
    #         f.write(source)
    #     log.info("已注入公共评测模块: %s", dest)
    # else:
    #     log.warning("未找到 copaw_eval.py 源码，跳过注入")

    # 2. 写入 conftest.py（仅当 task 未自带 conftest.py 时）
    conftest_path = os.path.join(test_dir, "conftest.py")
    if not os.path.exists(conftest_path):
        with open(conftest_path, "w", encoding="utf-8") as f:
            f.write(_CONFTEST_TEMPLATE)
        log.info("已注入公共 conftest.py: %s", conftest_path)


def _extract_task_description(instruction_text: str) -> str:
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
        return "\n\n".join(result_parts)
    return instruction_text


def _load_task_yaml(task_dir: str) -> dict | None:
    """Load task.yaml if present, return parsed dict or None."""
    yaml_path = os.path.join(task_dir, "task.yaml")
    if not os.path.isfile(yaml_path):
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_image_local_path(image_url: str, task_dir: str = "") -> str:
    """将 task.yaml 中的图片路径解析为容器内的绝对路径。

    解析顺序：
      1. 已经是绝对路径且文件存在 → 直接返回
      2. 若为绝对路径但文件不存在，提取 workspace 前缀后的相对部分继续查找
      3. deploy 后的 workspace 路径 → /app/working/workspaces/default/local_files/...
      4. task 下载目录中的原始路径 → _SCRIPT_DIR/environment/data/local_files/...
      5. 均未命中 → 原样返回并告警
    """
    if os.path.isabs(image_url) and os.path.isfile(image_url):
        return image_url

    workspace = "/app/working/workspaces/default"

    rel_url = image_url
    if os.path.isabs(image_url):
        known_prefixes = [
            workspace + "/local_files/",
            workspace + "/",
            "/root/",
        ]
        for prefix in known_prefixes:
            if image_url.startswith(prefix):
                rel_url = image_url[len(prefix) :]
                break
        else:
            rel_url = os.path.basename(image_url)

    local_candidates = [
        os.path.join(workspace, "local_files", rel_url),
        os.path.join(workspace, rel_url),
        os.path.join(task_dir, "environment", "data", "local_files", rel_url),
        os.path.join(task_dir, "environment", "data", rel_url),
        os.path.join(task_dir, rel_url),
    ]
    for path in local_candidates:
        if os.path.isfile(path):
            log.info("图片本地路径: %s", path)
            return path

    log.warning("无法解析图片路径: %s (尝试了 rel=%s)", image_url, rel_url)
    return image_url


def _build_agent_content(
    task_config: dict,
    task_dir: str = "",
) -> list[dict] | None:
    """从 task.yaml 的 messages 构建 CoPaw Agent API 所需的 content blocks。

    支持 text / image 等多模态内容块。
    image_url 会自动解析为容器内本地绝对路径。
    返回 None 表示 YAML 中没有有效 messages。
    """
    messages = task_config.get("messages", [])
    if not messages:
        return None

    user_msg = next((m for m in messages if m.get("role") == "user"), None)
    if not user_msg:
        return None

    content_blocks: list[dict] = []
    for item in user_msg.get("content", []):
        item_type = item.get("type", "")
        if item_type == "text":
            content_blocks.append(
                {
                    "type": "text",
                    "text": item["text"],
                    "status": "created",
                }
            )
        elif item_type == "image":
            raw_url = item.get("image_url", "")
            if isinstance(raw_url, dict):
                raw_url = raw_url.get("url", "")
            local_path = _resolve_image_local_path(raw_url, task_dir=task_dir)
            content_blocks.append(
                {
                    "type": "image",
                    "image_url": local_path,
                    "status": "created",
                }
            )
        elif item_type == "image_url":
            raw_url = item.get("image_url", {})
            if isinstance(raw_url, dict):
                raw_url = raw_url.get("url", "")
            local_path = _resolve_image_local_path(raw_url, task_dir=task_dir)
            content_blocks.append(
                {
                    "type": "image",
                    "image_url": local_path,
                    "status": "created",
                }
            )
        else:
            content_blocks.append({**item, "status": "created"})

    return content_blocks or None


def _extract_text_from_content(content_blocks: list[dict]) -> str:
    """从 content blocks 中提取纯文本部分，用于日志和 summary。"""
    return " ".join(
        block["text"]
        for block in content_blocks
        if block.get("type") == "text" and block.get("text")
    )


def deploy_environment(extract_dir: str) -> None:
    """将 environment/ 下的所有文件部署到 home。

    若 environment/ 下存在 config/、data/、skills/ 子目录：
      - config/ 下的文件覆盖复制到 home
      - data/   下的文件移动到 home
      - skills/ 下的文件复制到 home/skills/，并自动生成 skill.json manifest
    否则，直接将 environment/ 下的所有文件移动到 home。
    """
    env_dir = os.path.join(extract_dir, "environment")
    if not os.path.isdir(env_dir):
        log.warning("environment/ 目录不存在于解压目录: %s", env_dir)
        return

    home = "/app/working/workspaces/default"
    config_dir = os.path.join(env_dir, "config")
    data_dir = os.path.join(env_dir, "data")
    skills_dir = os.path.join(env_dir, "skills")

    has_config = os.path.isdir(config_dir)
    has_data = os.path.isdir(data_dir)
    has_skills = os.path.isdir(skills_dir)

    if has_config or has_data or has_skills:
        if has_config:
            shutil.copytree(config_dir, home, dirs_exist_ok=True)
            log.info("deploy config: %s -> %s", config_dir, home)
        if has_data:
            shutil.copytree(data_dir, home, dirs_exist_ok=True)
            log.info("deploy data: %s -> %s", data_dir, home)
        if has_skills:
            _deploy_skills(skills_dir, home)
    else:
        shutil.copytree(env_dir, home, dirs_exist_ok=True)
        log.info("deploy: %s -> %s", env_dir, home)


def _deploy_skills(skills_dir: str, workspace: str) -> None:
    """将 environment/skills/ 下的 skill 目录部署到 workspace，并生成 skill.json。

    每个包含 SKILL.md 的子目录被视为一个 skill，拷贝到 workspace/skills/ 下。
    同时自动生成 workspace/skill.json manifest，将所有 skill 标记为 enabled。
    如果 workspace 下已存在 skill.json，则合并新 skill 到现有 manifest 中。
    """
    dest_skills = os.path.join(workspace, "skills")
    os.makedirs(dest_skills, exist_ok=True)

    deployed = []
    for name in sorted(os.listdir(skills_dir)):
        src = os.path.join(skills_dir, name)
        if not os.path.isdir(src):
            continue
        skill_md = os.path.join(src, "SKILL.md")
        if not os.path.isfile(skill_md):
            log.warning("skip skill dir without SKILL.md: %s", name)
            continue
        dest = os.path.join(dest_skills, name)
        shutil.copytree(src, dest, dirs_exist_ok=True)
        deployed.append(name)
        log.info("deploy skill: %s -> %s", name, dest)

    if not deployed:
        return

    manifest_path = os.path.join(workspace, "skill.json")
    manifest = {
        "schema_version": "workspace-skill-manifest.v1",
        "version": 1,
        "skills": {},
    }
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    skills = manifest.setdefault("skills", {})
    for name in deployed:
        if name in skills:
            continue
        description = ""
        skill_md = os.path.join(dest_skills, name, "SKILL.md")
        try:
            with open(skill_md, "r", encoding="utf-8") as f:
                content = f.read()
            for line in content.splitlines():
                if line.startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip("'\"")
                    break
        except OSError:
            pass
        skills[name] = {
            "enabled": True,
            "channels": ["all"],
            "source": "benchmark",
            "metadata": {"name": name, "description": description},
            "requirements": {"require_bins": [], "require_envs": []},
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log.info("skill manifest written: %s (%d skills)", manifest_path, len(deployed))


def call_agent(
    url: str,
    user_input: str,
    session_id: str,
    user_id: str,
    provider_base_url: str,
    provider_model_id: str,
) -> None:
    if isinstance(user_input, str):
        content = [{"type": "text", "text": user_input, "status": "created"}]
    else:
        content = user_input

    payload = {
        "input": [
            {
                "role": "user",
                "type": "message",
                "content": content,
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
    # log.info("agent response: %s", response.text)

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


def parse_structured_trajectory(session_data: dict) -> list:
    """
    从 session JSON 的 _model_trajectory 中解析出结构化的 trajectory:
    [{"step": int, "thinking": str, "content": str, "tool_calls": [...], "tool_results": [...]}, ...]

    字段语义:
      - thinking: 模型内部推理 (response 中 type="thinking" 块)
      - content:  呈现给用户的文本 (response 中 type="text" 块)
      - tool_calls / tool_results: 工具调用与结果

    注意: messages 中的 reasoning_content 是**上一轮**的历史，不用于当前步骤。
    response 本身完全自包含当前步骤的 thinking + text + tool_use。

    各模型格式差异:
      - glm-5 / qwen3.5-plus / MiniMax: response 中有 thinking 块 + text 块
      - kimi-k2.5: response 中无 thinking 块，仅 text + tool_use
      - 训练模型 (local-4b 等): response 中无 thinking 块，推理内容放在 text 块中
    """
    raw_trajectory = session_data.get("agent", {}).get("_model_trajectory", [])
    if not raw_trajectory:
        return []

    trajectory = []
    for i, entry in enumerate(raw_trajectory):
        next_entry = raw_trajectory[i + 1] if i + 1 < len(raw_trajectory) else None
        response = entry.get("response", [])

        content = ""
        thinking = ""
        tool_calls = []

        if isinstance(response, str):
            content = response
        elif isinstance(response, list):
            for item in response:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "thinking":
                        thinking = item.get("thinking", "")
                    elif item_type == "text":
                        content = item.get("text", "")
                    elif item_type == "tool_use":
                        tool_calls.append(
                            {
                                "name": item.get("name", ""),
                                "input": item.get("input", {}),
                            }
                        )

        tool_results = []
        if next_entry and tool_calls:
            next_messages = next_entry.get("messages", [])
            current_messages = entry.get("messages", [])
            new_msg_count = len(next_messages) - len(current_messages)
            if new_msg_count >= 1:
                for msg in next_messages[-new_msg_count:]:
                    if msg.get("role") == "tool":
                        tool_content = msg.get("content", "")
                        if isinstance(tool_content, str):
                            tool_results.append(tool_content[:500])
                        elif isinstance(tool_content, list):
                            for c in tool_content:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    tool_results.append(c.get("text", "")[:500])

        trajectory.append(
            {
                "step": i,
                "thinking": thinking,
                "content": content,
                "tool_calls": tool_calls,
                "tool_results": tool_results,
            }
        )

    return trajectory


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


def _strip_ansi_codes(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _parse_grader_results(stdout: str) -> list[dict]:
    """从 pytest stdout 中提取 grader 结果，兼容 v1/v2 两种输出格式。

    v1: [label] score=X, reason=...
    v2: [label] (确定性) score=X, reason=...
        [label] (LLM, 非确定性) trial_scores=[...], selected=X, reason=...
    """
    results = []

    # v2 确定性: [label] (确定性) score=X, reason=...
    for m in re.finditer(
        r"^\[(.+?)\]\s+\(确定性\)\s+score=([\d.]+),\s+reason=(.*?)(?:\n|$)",
        stdout,
        re.MULTILINE,
    ):
        results.append(
            {
                "label": m.group(1),
                "score": float(m.group(2)),
                "reason": m.group(3).strip(),
                "type": "deterministic",
            }
        )

    # v2 非确定性: [label] (LLM, 非确定性) trial_scores=[...], (errors=N,) (high_variance=true,) selected=X, reason=...
    for m in re.finditer(
        r"^\[(.+?)\]\s+\(LLM, 非确定性\)\s+trial_scores=\[([^\]]*)\]"
        r"(?:,\s*errors=(\d+))?"
        r"(?:,\s*high_variance=(true|false))?"
        r",\s+selected=([\d.]+),\s+reason=(.*?)(?:\n|$)",
        stdout,
        re.MULTILINE,
    ):
        trial_scores = [float(s.strip()) for s in m.group(2).split(",") if s.strip()]
        entry = {
            "label": m.group(1),
            "score": float(m.group(5)),
            "reason": m.group(6).strip(),
            "type": "llm_grader",
            "trial_scores": trial_scores,
            "trial_errors": int(m.group(3)) if m.group(3) else 0,
        }
        if m.group(4) == "true":
            entry["high_variance"] = True
        results.append(entry)

    # v1 兼容: [label] score=X, reason=... (前面没有括号修饰)
    # 只匹配尚未被 v2 规则捕获的行
    v2_labels = {r["label"] for r in results}
    for m in re.finditer(
        r"^\[(.+?)\]\s+score=([\d.]+),\s+reason=(.*?)(?:\n|$)",
        stdout,
        re.MULTILINE,
    ):
        label = m.group(1)
        if label not in v2_labels:
            results.append(
                {
                    "label": label,
                    "score": float(m.group(2)),
                    "reason": m.group(3).strip(),
                }
            )

    # GRADER_ERROR（v1/v2 格式相同）
    for m in re.finditer(
        r"^\[(.+?)\]\s+GRADER_ERROR:\s+(.*?)(?:\n|$)",
        stdout,
        re.MULTILINE,
    ):
        results.append(
            {
                "label": m.group(1),
                "score": None,
                "error": m.group(2).strip(),
            }
        )

    for r in results:
        if "仅记录" in (r.get("label") or ""):
            r["record_only"] = True

    return results


def _clean_pytest_details(stdout: str) -> str:
    """精简 pytest 输出，去掉平台信息、插件列表等噪音，保留测试结果。"""
    lines = stdout.splitlines()
    cleaned: list[str] = []
    skip_patterns = [
        "platform linux",
        "cachedir:",
        "rootdir:",
        "configfile:",
        "plugins:",
        "asyncio: mode=",
        "collecting ...",
    ]
    for line in lines:
        if any(p in line for p in skip_patterns):
            continue
        # 跳过 openjudge 内部日志行
        if "openjudge." in line and "INFO" in line:
            continue
        cleaned.append(line)
    # 合并连续空行
    result: list[str] = []
    for line in cleaned:
        if line.strip() == "" and result and result[-1].strip() == "":
            continue
        result.append(line)
    return "\n".join(result).strip()


def parse_pytest_result(stdout: str) -> dict:
    """从 pytest 输出中解析测试结果，返回 total/passed/failed/skipped/errors/score。

    从 pytest 末尾汇总行（如 "4 passed, 1 failed in 5.2s"）提取计数。
    使用 findall 取最后一个匹配，避免误匹配测试名中的数字（如 test_keyword_0 PASSED）。
    """
    result = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "score": 0.0,
    }

    patterns = [
        (r"(\d+)\s+passed", "passed"),
        (r"(\d+)\s+failed", "failed"),
        (r"(\d+)\s+skipped", "skipped"),
        (r"(\d+)\s+error", "errors"),
    ]

    for pattern, key in patterns:
        matches = re.findall(pattern, stdout, re.IGNORECASE)
        if matches:
            result[key] = int(matches[-1])

    result["total"] = result["passed"] + result["failed"] + result["skipped"] + result["errors"]

    denominator = result["passed"] + result["failed"] + result["errors"]
    if denominator > 0:
        result["score"] = round(result["passed"] / denominator * 100, 1)

    return result


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _export_screenshots(session_data: dict) -> list[str]:
    """从 session 中导出 Agent 截取的屏幕/浏览器截图。

    包含：
      - desktop_screenshot 工具的输出（桌面截图）
      - browser_use action=screenshot 的输出（浏览器页面截图）
    排除：
      - view_image（仅查看已有图片，不是 Agent 生成的截图）
      - browser_use 的其他 action（open/click/snapshot 等）
    """
    exported: list[str] = []
    screenshots_dir = os.path.join(_SCRIPT_DIR, "screenshots")

    paths: list[str] = []
    for turn in session_data.get("agent", {}).get("memory", {}).get("content", []):
        if not turn:
            continue
        for msg in turn:
            if not isinstance(msg, dict):
                continue
            for block in msg.get("content", []):
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_result":
                    continue
                tool_name = block.get("name", "")
                out = block.get("output", block.get("content", []))

                if tool_name == "desktop_screenshot":
                    _collect_paths_from_tool_output(out, paths)

                elif tool_name == "browser_use":
                    _collect_browser_screenshot_paths(out, paths)

    seen: set[str] = set()
    for p in paths:
        if p in seen or not os.path.isfile(p):
            continue
        seen.add(p)
        os.makedirs(screenshots_dir, exist_ok=True)
        dest = os.path.join(screenshots_dir, os.path.basename(p))
        if os.path.exists(dest):
            base, ext = os.path.splitext(os.path.basename(p))
            dest = os.path.join(screenshots_dir, f"{base}_{len(seen)}{ext}")
        try:
            shutil.copy2(p, dest)
            exported.append(dest)
            log.info("截图已导出: %s → %s", p, dest)
        except Exception as e:
            log.warning("截图导出失败: %s → %s: %s", p, dest, e)

    if exported:
        log.info("共导出 %d 张截图到 %s", len(exported), screenshots_dir)
        zip_path = os.path.join(_SCRIPT_DIR, "screenshots.zip")
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", screenshots_dir)
        log.info("截图 zip 已生成: %s", zip_path)
    return exported


def _collect_paths_from_tool_output(out: list | str, paths: list[str]) -> None:
    """从 tool_result output 中提取文件路径。"""
    if isinstance(out, str):
        m = re.search(r'"path"\s*:\s*"([^"]+)"', out)
        if m:
            paths.append(m.group(1))
        return
    if not isinstance(out, list):
        return
    for item in out:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            m = re.search(r'"path"\s*:\s*"([^"]+)"', item.get("text", ""))
            if m:
                paths.append(m.group(1))
        elif item.get("type") == "image":
            url = item.get("source", {}).get("url", "")
            if url:
                paths.append(url.replace("file://", ""))


def _collect_browser_screenshot_paths(out: list | str, paths: list[str]) -> None:
    """从 browser_use tool_result 中提取截图路径（仅 screenshot action）。

    browser_use screenshot 的 result 格式:
      {"ok": true, "message": "Screenshot saved to ...", "path": "/path/to/file.png"}
    其他 action（open/click/snapshot 等）没有指向图片文件的 path 字段。
    """
    text = ""
    if isinstance(out, str):
        text = out
    elif isinstance(out, list):
        text = " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in out
        )
    if not text:
        return

    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        data = None

    if isinstance(data, dict):
        msg = data.get("message", "")
        path = data.get("path", "")
        if path and "screenshot" in msg.lower():
            ext = os.path.splitext(path)[1].lower()
            if ext in _IMAGE_EXTS:
                paths.append(path)


_WORKSPACE_EXPORT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per file
_WORKSPACE_EXPORT_MAX_TOTAL_SIZE = 200 * 1024 * 1024  # 200MB total
_WORKSPACE_EXPORT_SKIP_DIRS = {
    "sessions",
    ".cache",
    "__pycache__",
    "node_modules",
    ".git",
    "browser",  # Chrome/Chromium profile data (user_data, cache, etc.)
    "tool_result",  # CoPaw 工具调用中间结果
    "local_files",  # 任务输入文件（非 Agent 产出）
}
_WORKSPACE_EXPORT_SKIP_FILES = {
    "skill.json",  # CoPaw skill 配置
    "chats.json",  # CoPaw 会话记录
    "BOOTSTRAP.md",  # CoPaw 引导文件
    "PROFILE.md",  # CoPaw 用户 profile
}
_WORKSPACE_EXPORT_SKIP_EXTS = {".pyc", ".pyo", ".so", ".o", ".a", ".dylib"}


def _export_workspace_files(baseline_snapshot: dict[str, float] | None = None) -> list[str]:
    """扫描 workspace 目录，导出 Agent 运行期间新增或修改的文件。

    Args:
        baseline_snapshot: setup.sh 执行后、Agent 运行前的 workspace 文件快照
                          {relative_path: mtime}，为 None 则导出全部文件。

    Returns:
        导出的文件路径列表。
    """
    workspace = "/app/working/workspaces/default"
    if not os.path.isdir(workspace):
        return []

    export_dir = os.path.join(_SCRIPT_DIR, "workspace_files")
    exported: list[str] = []
    total_size = 0

    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _WORKSPACE_EXPORT_SKIP_DIRS]
        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, workspace)

            if fname in _WORKSPACE_EXPORT_SKIP_FILES:
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in _WORKSPACE_EXPORT_SKIP_EXTS:
                continue

            try:
                stat = os.stat(full_path)
            except OSError:
                continue

            if stat.st_size > _WORKSPACE_EXPORT_MAX_FILE_SIZE:
                log.info("跳过大文件: %s (%d bytes)", rel_path, stat.st_size)
                continue

            if baseline_snapshot is not None:
                old_mtime = baseline_snapshot.get(rel_path)
                if old_mtime is not None and stat.st_mtime <= old_mtime:
                    continue

            if total_size + stat.st_size > _WORKSPACE_EXPORT_MAX_TOTAL_SIZE:
                log.warning("workspace 导出总量已达上限，停止收集")
                break

            dest = os.path.join(export_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                shutil.copy2(full_path, dest)
                exported.append(rel_path)
                total_size += stat.st_size
            except Exception as e:
                log.warning("workspace 文件导出失败: %s: %s", rel_path, e)

    if exported:
        log.info("workspace 文件导出: %d 个文件, %.1f KB", len(exported), total_size / 1024)
        zip_path = os.path.join(_SCRIPT_DIR, "workspace_files.zip")
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", export_dir)
        log.info("workspace_files.zip 已生成: %s", zip_path)
    else:
        log.info("workspace 无新增/修改文件，跳过导出")

    return exported


def _snapshot_workspace() -> dict[str, float]:
    """对 workspace 目录做快照，记录所有文件的 mtime。"""
    workspace = "/app/working/workspaces/default"
    snapshot: dict[str, float] = {}
    if not os.path.isdir(workspace):
        return snapshot
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _WORKSPACE_EXPORT_SKIP_DIRS]
        for fname in files:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, workspace)
            try:
                snapshot[rel_path] = os.stat(full_path).st_mtime
            except OSError:
                pass
    return snapshot


def _save_summary(summary: dict) -> None:
    """将 summary JSON 写到文件。"""
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("summary 已保存到: %s", SUMMARY_PATH)


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

        # 对 workspace 做快照，用于后续导出 diff
        workspace_baseline = _snapshot_workspace() if args.evaluation else None

        # Step 3: 构建 agent 输入（优先 task.yaml 多模态，兼容 instruction.md）
        task_config = _load_task_yaml(_SCRIPT_DIR)
        content_blocks = (
            _build_agent_content(task_config, task_dir=_SCRIPT_DIR) if task_config else None
        )

        if content_blocks:
            agent_input: str | list[dict] = content_blocks
            user_input = _extract_text_from_content(content_blocks)
            log.info(
                "从 task.yaml 构建多模态消息 (%d blocks), 文本长度: %d 字符", len(content_blocks), len(user_input)
            )
        else:
            instruction_path = os.path.join(_SCRIPT_DIR, "instruction.md")
            if not os.path.exists(instruction_path):
                log.error("instruction.md 不存在: %s", instruction_path)
                sys.exit(1)
            full_instruction = read_file(instruction_path)
            user_input = _extract_task_description(full_instruction)
            agent_input = user_input
            log.info("读取 instruction.md，query 长度: %d 字符", len(user_input))

        # Step 4: 调用 agent API
        log.info("调用 agent API ...")
        t_start = time.time()
        call_agent(
            args.url.strip("/"),
            agent_input,
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

        # 将原始 session 文件复制到 /root/session.json 方便下载
        session_export_path = os.path.join(_SCRIPT_DIR, "session.json")
        shutil.copy2(session_file, session_export_path)
        log.info("原始 session 已导出到: %s", session_export_path)

        final_text = extract_final_text(session_data)
        log.info(f"Agent 最终回复文本:\n{final_text}\n")

        trajectories = extract_trajectories(session_data)

        if not args.evaluation:
            export_training_data(args.task_id, trajectories)
            return

        structured_trajectory = parse_structured_trajectory(session_data)

        # Step 5.5: 导出 session 中截到的图片到结果目录
        _export_screenshots(session_data)

        # Step 5.6: 导出 workspace 中新增/修改的文件
        _export_workspace_files(baseline_snapshot=workspace_baseline)

        # Step 6: 保存 traj 到 judge 所在的目录 /traj.json
        traj_path = os.path.join(test_dir, "traj.json")
        with open(traj_path, "w", encoding="utf-8") as f:
            json.dump(trajectories, f, ensure_ascii=False, indent=2)
        log.info("trajectory 已保存到: %s", traj_path)

        # Step 7: 打包日志文件
        log.info("打包日志文件...")
        shutil.make_archive(
            os.path.join(_SCRIPT_DIR, "tests"), "zip", os.path.join(_SCRIPT_DIR, "tests")
        )

        # Step 7.5: 注入公共评测模块 copaw_eval.py 到 tests/ 目录
        _inject_shared_eval_module(test_dir)

        # Step 8: 运行 tests/test_outputs.py 并输出结果（控制台保持原样）
        test_script = os.path.join(test_dir, "test_outputs.py")
        if not os.path.exists(test_script):
            log.error("tests/test_outputs.py 不存在: %s", test_script)
            sys.exit(1)
        log.info("运行测试: %s", test_script)
        test_env = os.environ.copy()
        test_env["SESSION_FILE"] = session_file
        # CoPaw 工作区路径，供 test_outputs 中 INPUT_DIR/PROFILE_PATH/SESSION_FOLDER 等使用
        workspace_dir = "/app/working/workspaces/default"
        test_env.setdefault("INPUT_DIR", workspace_dir)
        test_env.setdefault("PROFILE_PATH", os.path.join(workspace_dir, "PROFILE.md"))
        test_env.setdefault("SESSION_FOLDER", os.path.join(workspace_dir, "sessions"))

        # 自动设置 OUTPUT_FILE / OUTPUT_DIR：从 test_outputs.py 中解析默认路径，
        # 提取文件名后映射到 workspace 下的 output 目录
        with open(test_script, "r", encoding="utf-8") as f:
            test_content = f.read()
        if "OUTPUT_FILE" in test_content:
            m = re.search(
                r"""OUTPUT_FILE\s*=.*?(?:["'])((?:~/|/root/|/app/working/)output/[^"']+)(?:["'])""",
                test_content,
            )
            if m:
                output_filename = os.path.basename(m.group(1))
                test_env.setdefault(
                    "OUTPUT_FILE",
                    os.path.join(workspace_dir, "output", output_filename),
                )
                log.info("OUTPUT_FILE=%s", test_env["OUTPUT_FILE"])
        if "OUTPUT_DIR" in test_content and "OUTPUT_FILE" not in test_content:
            test_env.setdefault("OUTPUT_DIR", os.path.join(workspace_dir, "output"))
            log.info("OUTPUT_DIR=%s", test_env["OUTPUT_DIR"])
        # -s/--capture=no：否则「先 print 再 pytest.skip」的用例里，评分行留在
        # pytest 内部 capture，不会进入本进程的 stdout，run.py 无法解析
        # record_only_grader_scores / grader_results。
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-v",
                "-rP",
                "-s",
                test_script,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=_SCRIPT_DIR,
            env=test_env,
        )
        # sys.stdout.write(result.stdout or "")
        log.info("测试输出:\n%s", result.stdout)
        log.info("测试错误输出:\n%s", result.stderr)
        log.info("测试退出码: %d", result.returncode)

        # Step 9: 构建 summary JSON 写到文件
        stdout_clean = _strip_ansi_codes(result.stdout)
        test_result = parse_pytest_result(stdout_clean)
        grader_results = _parse_grader_results(stdout_clean)

        deterministic = [r for r in grader_results if r.get("type") == "deterministic"]
        llm_grader = [
            r for r in grader_results if r.get("type") == "llm_grader" and not r.get("record_only")
        ]
        record_only_graders = [r for r in grader_results if r.get("record_only")]
        legacy = [r for r in grader_results if "type" not in r and r.get("score") is not None]

        eval_result = {
            "status": "passed" if result.returncode == 0 else "failed",
            "score": test_result["score"],
            "tests": {
                "total": test_result["total"],
                "passed": test_result["passed"],
                "failed": test_result["failed"],
                "skipped": test_result["skipped"],
                "errors": test_result["errors"],
            },
            "grader_results": grader_results or None,
            "deterministic_scores": deterministic or None,
            "llm_grader_scores": llm_grader or None,
            # 始终输出数组，便于下游展示；无仅记录项时为 []
            "record_only_grader_scores": record_only_graders if record_only_graders else [],
            "legacy_scores": legacy or None,
            "details": _clean_pytest_details(stdout_clean) or None,
        }

        _save_summary(
            {
                "run_name": args.task_id,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_url": args.url,
                "summary": {
                    "total_tasks": 1,
                    "completed": 1,
                    "failed": 0 if result.returncode == 0 else 1,
                    "avg_score": eval_result["score"],
                },
                "tasks": [
                    {
                        "session_id": args.session_id,
                        "final_text": final_text,
                        "status": "completed",
                        "duration_seconds": duration_seconds,
                        "steps": len(structured_trajectory),
                        "query": user_input,
                        "trajectory": structured_trajectory,
                        "evaluation": eval_result,
                        "task": args.task_id,
                    }
                ],
            }
        )

        sys.exit(result.returncode)

    except SystemExit:
        raise
    except Exception as e:
        log.exception("运行异常: %s", e)
        if args.evaluation:
            _save_summary(
                {
                    "run_name": getattr(args, "task_id", "unknown"),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "base_url": getattr(args, "url", ""),
                    "summary": {"total_tasks": 1, "completed": 0, "failed": 1, "avg_score": 0},
                    "tasks": [
                        {
                            "session_id": getattr(args, "session_id", ""),
                            "final_text": "",
                            "status": "error",
                            "duration_seconds": 0,
                            "steps": 0,
                            "query": "",
                            "trajectory": [],
                            "evaluation": {"status": "error", "score": 0.0, "error": str(e)},
                            "task": getattr(args, "task_id", "unknown"),
                        }
                    ],
                }
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
