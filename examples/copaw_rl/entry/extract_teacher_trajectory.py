#!/usr/bin/env python3

"""
python3 /Users/maixinji/code/CoPaw-Pro/03_copaw_client_multi.py


# 中断后，带指定目录恢复运行（Resume 自动跳过已完成的）
COPAW_RESUME=1 COPAW_TRAJ_DIR=/Users/maixinji/code/traj_MiniMax-M2.5_20260318_143719_0318 \
python3 03_copaw_client_multi.py
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from urllib import request as urllib_request

spec = importlib.util.spec_from_file_location(
    "sandbox_utils",
    os.path.join(os.path.dirname(__file__), "..", "workflows", "sandbox_utils.py"),
)
sandbox_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sandbox_utils)
get_or_create_sandbox = sandbox_utils.get_or_create_sandbox
run_teacher_workflow = sandbox_utils.run_teacher_workflow


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ── 拒绝采样：LLM 判断配置 ──────────────────────────────────────────────────
MAX_REJECTION_SAMPLES = 3
JUDGE_MODEL = "qwen-plus"
JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
JUDGE_SYSTEM_PROMPT = (
    "你是 CoPaw 智能体训练数据的质量评审员。CoPaw 是个人 AI 助手，"
    "trajectory 来自它在沙盒里完成各种任务（文档/PDF 解析、知识问答、"
    "本地文件查询与整理、定时任务设置、人格 onboarding 引导等），"
    "可调用 shell、读写文件、检索等工具。\n"
    "你将看到该样本的【首条用户意图 first_user_message】、"
    "【最后一轮模型回复 last_response】、以及一份"
    "【过程摘要 trajectory_summary】（含工具调用次数、报错次数、最后一条工具输出尾段等），"
    "需要判断这条样本是否可作为优质 SFT 数据被采纳。\n"
    "\n"
    "【关键原则：评判结果，不评判过程】\n"
    "- trajectory_summary 中的 tool_error_count / 工具调用次数只是过程信号，"
    "智能体在多步任务中试错、踩坑、再换方案是**健康且常见**的行为，"
    "不要仅因为有工具报错、报错次数多、调用了很多工具就判失败。\n"
    "- 唯一关注的是：模型最终是否走出了试错、并在 last_response 中向用户给出了可交付结果。\n"
    "- 也就是说：高 tool_error_count + 最终给出可交付结果 ⇒ 仍然判 success=true；"
    "tool_error_count=0 但最后只是过程性表态 / 没有交付 ⇒ 仍然判 success=false。\n"
    "\n"
    "【判定为完成 success=true 的情形】\n"
    "1) last_response 已直接交付用户首条意图的核心产出（答案、结论、文件内容摘要、操作确认等），"
    "无论中间踩了多少坑。\n"
    "2) 用户意图本身强依赖人类输入（如个人偏好、私人密码、身份信息等），"
    "模型在穷尽上下文线索后才合理向用户回问，且问题精准——视为已尽责完成本轮。\n"
    "3) 风格啰嗦/绕路/部分溢出，只要主要诉求已闭环，仍判完成。\n"
    "\n"
    "【判定为未完成 success=false 的情形，命中任一即 false】\n"
    "1) 声称缺少文件/资料但未在容器内尝试查找——CoPaw 任务环境必然预置所需文件，"
    "找不到等同于模型能力不足（注意区分：尝试找过但确实不存在 ≠ 此条）。\n"
    "2) 工具一路报错到最后仍没有恢复，last_response 只剩错误说明 / 中断陈述，无可交付结果"
    "（判断点是“最终是否恢复”，不是“中间有没有报错”）。\n"
    "3) 信息其实充足却向用户发起多余反问（如让用户重述任务里已明示的内容）。\n"
    "4) 编造未在工具结果中出现过的事实数据（具体数字、引用、文件名等），或答非所问。\n"
    "5) 仅给出过程性表态（如“我将开始执行…”、“收到，稍后处理”）而没有最终交付。\n"
    "6) 对明确且无害的合理请求作出无理拒绝。\n"
    "\n"
    "【输出格式 严格执行】\n"
    "只输出一个 JSON 对象，键顺序必须先 reason 后 success，"
    "不要 markdown、不要多余文本：\n"
    '{"reason": "<不超过 60 字中文，给出具体判定依据>", "success": true 或 false}'
)


# ── 拒绝采样：trajectory 摘要 / judge 客户端 ─────────────────────────────
# 喂给 judge 时单字段最大长度，防止 last_response / 工具输出爆 qwen-plus 上下文。
_JUDGE_MAX_FIELD_CHARS = 4000

# judge 接口自身的重试参数（指数退避：5s, 10s, 20s ...）
_JUDGE_HTTP_RETRIES = 3
_JUDGE_HTTP_BACKOFF_BASE = 5.0


def has_tool_calls(traj: list) -> bool:
    """检查 trajectory 中是否存在 role 为 tool 的消息。"""
    for entry in traj:
        if not isinstance(entry, dict):
            continue
        for msg in entry.get("messages", []):
            if isinstance(msg, dict) and msg.get("role") == "tool":
                return True
    return False


def _clamp_text(value, n: int = _JUDGE_MAX_FIELD_CHARS) -> str:
    """把任意结构压成字符串并截断，保留首尾各一半 + 中间 truncated 提示。"""
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            value = str(value)
    if len(value) <= n:
        return value
    half = max(1, n // 2)
    return value[:half] + f"\n...[truncated {len(value) - n} chars]...\n" + value[-half:]


def _extract_first_user_message(trajectory: list) -> str:
    if not trajectory or not isinstance(trajectory[0], dict):
        return ""
    for m in trajectory[0].get("messages", []):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "\n".join(
                item.get("text", "")
                for item in c
                if isinstance(item, dict) and item.get("type") == "text"
            )
        break
    return ""


def _summarize_trajectory(trajectory: list) -> dict:  # noqa: C901
    """从 trajectory 抽取关键信号供 judge 参考：工具调用次数、报错次数、最后一条
    tool 输出尾段等。比只看 last_response 更能识别"找不到文件 / 工具报错没恢复"。
    """
    tools_used: list[str] = []
    tool_call_count = 0
    tool_error_count = 0
    last_tool_name = ""
    last_tool_output = ""
    for entry in trajectory or []:
        if not isinstance(entry, dict):
            continue
        for msg in entry.get("messages", []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    name = ((tc.get("function") or {}).get("name")) or tc.get("name") or ""
                    if name:
                        tool_call_count += 1
                        if name not in tools_used:
                            tools_used.append(name)
            elif role == "tool":
                name = msg.get("name") or ""
                content = msg.get("content", "")
                if isinstance(content, list):
                    parts: list[str] = []
                    for item in content:
                        if isinstance(item, dict):
                            t = item.get("text") or item.get("content") or ""
                            if isinstance(t, str):
                                parts.append(t)
                    content = "\n".join(parts)
                content_str = (
                    content
                    if isinstance(content, str)
                    else json.dumps(content, ensure_ascii=False, default=str)
                )
                lower = content_str.lower()
                if any(k in lower for k in ("traceback", "error", "exception", '"type": "error"')):
                    tool_error_count += 1
                if name:
                    last_tool_name = name
                last_tool_output = content_str
    return {
        "tool_call_count": tool_call_count,
        "tool_error_count": tool_error_count,
        "tools_used": tools_used[:20],
        "last_tool_name": last_tool_name,
        "last_tool_output_tail": _clamp_text(last_tool_output, _JUDGE_MAX_FIELD_CHARS // 2),
    }


def _llm_judge_sync(
    sample_id: str,
    first_user_message: str,
    last_response,
    traj_summary: dict,
) -> tuple[bool, str]:
    """同步版 LLM 判断，在 executor 中调用。内部 3 次重试 + 指数退避；
    全部失败时抛 RuntimeError，由调用方决定如何处理（保守判失败）。"""
    if last_response is None:
        return False, "trajectory为空或最后一轮无response"

    user_payload = {
        "sample_id": sample_id,
        "first_user_message": _clamp_text(first_user_message),
        "last_response": _clamp_text(last_response),
        "trajectory_summary": traj_summary,
    }
    payload = {
        "model": JUDGE_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }
    body_bytes = json.dumps(payload).encode("utf-8")
    api_key = os.environ.get("DASHSCOPE_API_KEY")

    last_exc: Exception | None = None
    for attempt in range(1, _JUDGE_HTTP_RETRIES + 1):
        try:
            req = urllib_request.Request(
                JUDGE_BASE_URL,
                data=body_bytes,
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
            return (
                bool(obj.get("success", False)),
                str(obj.get("reason", "")).strip(),
            )
        except Exception as exc:
            last_exc = exc
            if attempt < _JUDGE_HTTP_RETRIES:
                time.sleep(_JUDGE_HTTP_BACKOFF_BASE * (2 ** (attempt - 1)))

    raise RuntimeError(f"judge HTTP failed after {_JUDGE_HTTP_RETRIES} attempts: {last_exc}")


async def _llm_judge(sample_id: str, trajectory: list) -> tuple[bool, str]:
    """异步 LLM 判断 trajectory 是否成功。"""
    # 所有任务都要求模型至少调用一次工具去解决（包括纯知识题、bootstrap 等）。
    if not has_tool_calls(trajectory):
        return False, "trajectory中没有工具调用"

    first_user_message = _extract_first_user_message(trajectory)

    last_response = None
    if trajectory and isinstance(trajectory[-1], dict) and "response" in trajectory[-1]:
        last_response = trajectory[-1]["response"]

    traj_summary = _summarize_trajectory(trajectory)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        _llm_judge_sync,
        sample_id,
        first_user_message,
        last_response,
        traj_summary,
    )


def _strip_ansi_codes(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _parse_pytest_result(text: str) -> dict:
    """从 pytest 输出文本中提取 passed/failed 数量。

    只解析最后一行「===== … in X.XXs =====」样式的 pytest 汇总行。整段日志里
    若先用 ``re.search(r'(\\d+) failed')``，断言/报错中的子串（例如
    ``2144155449 failed to ...``）会被误当成失败用例数，汇总求和后会出现异常大的总数。

    解析失败统一返回空 dict ``{}``，让上层用 ``if stats:`` 区分"没拿到结果"
    和"真的全 0"。
    """
    if not (text and text.strip()):
        return {}

    text = _strip_ansi_codes(text)
    summary_line: str | None = None
    for line in text.splitlines():
        s = line.strip()
        if not (s.startswith("=") and s.endswith("=")):
            continue
        if not re.search(r"\bin\s+[\d.]+\s*s\b", s, re.IGNORECASE):
            continue
        inner = s.strip("=").strip()
        if inner:
            summary_line = s

    if summary_line is None:
        return {}

    inner = summary_line.strip("=").strip()
    m_time = re.search(r"\s+in\s+[\d.]+\s*s\s*$", inner, re.IGNORECASE)
    counts_part = inner[: m_time.start()] if m_time else inner

    result = {"passed": 0, "failed": 0, "total": 0}
    m = re.search(r"(\d+)\s+passed", counts_part, re.IGNORECASE)
    if m:
        result["passed"] = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", counts_part, re.IGNORECASE)
    if m:
        result["failed"] = int(m.group(1))

    result["total"] = result["passed"] + result["failed"]
    if result["total"] == 0:
        return {}
    return result


def _extract_run_text(run_outputs: list) -> str:
    """收集 run_sample 的所有 output 文本。pytest 失败时 env_sandbox 会把
    docker exec 的 stdout 丢进 stderr/error 段，遍历全部 output 才能拿到
    pytest 汇总行。"""
    if not run_outputs:
        return ""
    parts: list[str] = []
    for item in run_outputs:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str) and content:
            parts.append(content)
        elif isinstance(content, list):
            for sub in content:
                if isinstance(sub, dict):
                    sub_text = sub.get("text") or sub.get("content")
                    if isinstance(sub_text, str) and sub_text:
                        parts.append(sub_text)
    return "\n".join(parts)


def _print_tool_result(sample_id: str, step: str, result: dict, verbose: bool = False) -> None:
    outputs = result.get("outputs", [])
    if outputs and outputs[0].get("type") == "text":
        content = outputs[0].get("content", "").rstrip()
        if verbose and content:
            print(f"[{sample_id}] {step}:\n{content}")
        elif content:
            print(f"[{sample_id}] {step}: {content}")
        else:
            print(f"[{sample_id}] {step}: success")
        return

    # 有 error 类型输出时额外标注
    if outputs and outputs[0].get("type") == "error":
        print(f"[{sample_id}] {step} ERROR: {outputs[0].get('content', '')}")
        return

    print(f"[{sample_id}] {step}: {result}")


def _load_sample_ids_from_txt(sample_ids_path: Path) -> list[str]:
    if not sample_ids_path.is_file():
        raise FileNotFoundError(f"Sample ID txt not found: {sample_ids_path}")

    raw_lines = sample_ids_path.read_text(encoding="utf-8").splitlines()
    sample_ids: list[str] = []
    seen: set[str] = set()

    for line in raw_lines:
        sample_id = line.strip()
        if not sample_id or sample_id.startswith("#"):
            continue
        if sample_id not in seen:
            seen.add(sample_id)
            sample_ids.append(sample_id)

    if not sample_ids:
        raise ValueError(f"No valid sample IDs found in: {sample_ids_path}")

    return sample_ids


def _load_completed_trajectories(
    traj_dir: Path,
    valid_sample_ids: set[str],
) -> tuple[dict[str, object], list[str]]:
    completed: dict[str, object] = {}
    invalid_sample_ids: list[str] = []

    for path in sorted(traj_dir.glob("*.json")):
        sample_id = path.stem
        if sample_id not in valid_sample_ids:
            continue

        try:
            completed[sample_id] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            invalid_sample_ids.append(sample_id)

    return completed, invalid_sample_ids


def _rebuild_merged_jsonl(
    merged_jsonl_path: Path,
    completed_trajectories: dict[str, object],
) -> None:
    with merged_jsonl_path.open("w", encoding="utf-8") as f:
        for sample_id in sorted(completed_trajectories):
            f.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "trajectory": completed_trajectories[sample_id],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


async def _run_single_attempt(
    sample_id: str,
    model_name: str,
) -> tuple[str, list, dict]:
    """在一个独立沙箱中运行一次样本，返回 (file_content, trajectory, pytest_stats)。"""

    token = os.environ.get("E2B_API_KEY")
    domain = os.environ.get("E2B_DOMAIN")
    template = os.environ.get("E2B_TEMPLATE")
    sandbox, created = get_or_create_sandbox("", token, domain, template, logger)

    oss_config = {
        "access_key_id": os.environ.get("OSS_ACCESS_KEY_ID"),
        "access_key_secret": os.environ.get("OSS_ACCESS_KEY_SECRET"),
        "region": os.environ.get("OSS_REGION"),
        "endpoint": os.environ.get("OSS_ENDPOINT"),
        "bucket_name": os.environ.get("OSS_BUCKET_NAME"),
        "prefix": os.environ.get("OSS_PREFIX"),
    }
    dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")

    try:
        trajectory_file, trajectory, run_outputs = run_teacher_workflow(
            sandbox, sample_id, oss_config, dashscope_api_key, model_name, logger
        )
    finally:
        sandbox.kill()

    run_text = _extract_run_text(run_outputs)
    pytest_stats = _parse_pytest_result(run_text)
    if not pytest_stats:
        tail = run_text[-400:].replace("\n", " ⏎ ")
        print(f"[{sample_id}] pytest 汇总未解析到 (tail400): {tail}")

    return trajectory_file, trajectory, pytest_stats


async def _process_sample(
    index: int,
    total: int,
    sample_id: str,
    model_name: str,
    traj_dir: Path,
    merged_jsonl_path: Path,
    merged_jsonl_lock: asyncio.Lock,
) -> tuple[str, str | None, dict]:
    print(f"[{index}/{total}] running sample: {sample_id}")

    last_pytest_stats: dict = {}

    for attempt in range(1, MAX_REJECTION_SAMPLES + 1):
        try:
            file_content, trajectory, pytest_stats = await _run_single_attempt(
                sample_id, model_name
            )
        except Exception as exc:
            print(
                f"[{sample_id}] Attempt {attempt}/{MAX_REJECTION_SAMPLES} raised exception: {exc}"
            )
            if attempt < MAX_REJECTION_SAMPLES:
                continue
            # 所有尝试均以异常结束
            print(f"[{sample_id}] All {MAX_REJECTION_SAMPLES} attempts failed with exceptions")
            return sample_id, str(exc), {}

        last_pytest_stats = pytest_stats

        # ── LLM 判断是否执行成功 ──────────────────────────────────────────
        try:
            judge_ok, judge_reason = await _llm_judge(sample_id, trajectory)
        except Exception as judge_exc:
            # 判断接口异常（重试 3 次仍失败）保守判失败，避免污染数据集。
            judge_ok, judge_reason = False, f"LLM判断异常(保守视为失败): {judge_exc}"

        print(
            f"[{sample_id}] Attempt {attempt}/{MAX_REJECTION_SAMPLES} "
            f"-> judge={'✓' if judge_ok else '✗'} | {judge_reason}"
        )

        if judge_ok:
            # 判断成功，保存并退出
            output_path = traj_dir / f"{sample_id}.json"
            output_path.write_text(file_content)

            async with merged_jsonl_lock:
                with merged_jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {"sample_id": sample_id, "trajectory": trajectory},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(f"[{sample_id}] Trajectory saved (attempt {attempt})")
            return sample_id, None, pytest_stats

        # 判断失败，若还有机会则继续采样
        if attempt < MAX_REJECTION_SAMPLES:
            print(f"[{sample_id}] Retrying (attempt {attempt + 1}/{MAX_REJECTION_SAMPLES})...")

    # 所有采样均被 LLM 拒绝，不保存（拒绝采样）
    print(f"[{sample_id}] Rejected after {MAX_REJECTION_SAMPLES} attempts, trajectory not saved")
    return sample_id, f"rejected after {MAX_REJECTION_SAMPLES} attempts", last_pytest_stats


async def main() -> None:  # noqa: C901
    os.environ.setdefault("COPAW_SAMPLE_ID_FILE", "/mnt/chencheng.cc/CoPaw-Pro/train_samples.txt")
    os.environ.setdefault("COPAW_RESUME", "1")
    os.environ.setdefault("COPAW_CONCURRENCY", "60")

    _traj_dir = os.environ.get("COPAW_TRAJ_DIR", "")
    _resume = os.environ.get("COPAW_RESUME", "0")
    if _traj_dir and _resume == "1":
        _m = re.match(r"traj_(.+?)_\d{8}", Path(_traj_dir).name)
        model_name = _m.group(1) if _m else "qwen3.5-flash"
    else:
        model_name = "qwen3.5-flash"
    print("model_name:", model_name)
    # glm-5 MiniMax-M2.5 kimi-k2.5 vertex_ai.claude-opus-4-6 openai.gpt-5.4-2026-03-05 vertex_ai.gemini-3.1-pro-preview grok-4-1-fast-reasoning

    workspace_root = Path(__file__).resolve().parent.parent
    # script_dir = Path(__file__).resolve().parent
    import time

    _traj_dir_env = os.environ.get("COPAW_TRAJ_DIR", "")
    if _traj_dir_env:
        traj_dir = Path(_traj_dir_env)
    else:
        traj_dir = workspace_root / f"traj_{model_name}_{time.strftime('%Y%m%d_%H%M%S')}_0319"
    merged_jsonl_path = traj_dir / "all_traj.jsonl"
    # run_script_path = script_dir / "worker_patch" / "run.py"
    sample_ids_path = Path(
        os.environ.get("COPAW_SAMPLE_ID_FILE", str(workspace_root / "sample_ids.txt"))
    )

    concurrency = max(1, int(os.environ.get("COPAW_CONCURRENCY", "4")))
    resume_enabled = os.environ.get("COPAW_RESUME", "1").lower() not in {"0", "false", "no"}

    # if not run_script_path.is_file():
    #     raise FileNotFoundError(f"Run script not found: {run_script_path}")

    traj_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = _load_sample_ids_from_txt(sample_ids_path)
    random.shuffle(sample_ids)
    print(f"Loaded {len(sample_ids)} sample IDs (shuffled)")
    sample_id_set = set(sample_ids)

    if resume_enabled:
        completed_trajectories, invalid_sample_ids = _load_completed_trajectories(
            traj_dir, sample_id_set
        )
        if invalid_sample_ids:
            print("Ignoring invalid trajectory files and rerunning:", sorted(invalid_sample_ids))

        _rebuild_merged_jsonl(merged_jsonl_path, completed_trajectories)
        pending_sample_ids = [
            sample_id for sample_id in sample_ids if sample_id not in completed_trajectories
        ]

        print(
            f"Resume enabled: completed={len(completed_trajectories)} pending={len(pending_sample_ids)}"
        )
    else:
        merged_jsonl_path.write_text("", encoding="utf-8")
        pending_sample_ids = sample_ids
        print("Resume disabled: rerunning all samples")

    if not pending_sample_ids:
        print("All samples are already completed")
        return

    print(f"Running {len(pending_sample_ids)} samples with concurrency={concurrency}")

    succeeded = sorted(set(sample_ids) - set(pending_sample_ids))
    failed: list[str] = []
    test_stats: dict[str, dict] = {}  # sample_id -> pytest_stats
    merged_jsonl_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_process(index: int, sample_id: str) -> tuple[str, str | None, dict]:
        async with semaphore:
            return await _process_sample(
                index=index,
                total=len(sample_ids),
                sample_id=sample_id,
                model_name=model_name,
                traj_dir=traj_dir,
                merged_jsonl_path=merged_jsonl_path,
                merged_jsonl_lock=merged_jsonl_lock,
            )

    tasks = [
        asyncio.create_task(bounded_process(index, sample_id))
        for index, sample_id in enumerate(pending_sample_ids, start=1)
    ]

    for completed in asyncio.as_completed(tasks):
        sample_id, error, stats = await completed
        if stats:
            test_stats[sample_id] = stats
        if error is None:
            succeeded.append(sample_id)
        else:
            failed.append(sample_id)

    print(f"Finished. success={len(succeeded)} failed={len(failed)}")
    if failed:
        print("Failed samples:", failed)

    # ── 补跑：trajectory 为空 list 的样本 ──────────────────────────────────
    empty_traj_ids: list[str] = []
    if merged_jsonl_path.exists():
        with merged_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj.get("trajectory"), list) and len(obj["trajectory"]) == 0:
                    empty_traj_ids.append(obj["sample_id"])

    if empty_traj_ids:
        print(f"\n发现 {len(empty_traj_ids)} 个 trajectory 为空的样本，开始补跑: {empty_traj_ids}")

        retry_tasks = [
            asyncio.create_task(bounded_process(index, sample_id))
            for index, sample_id in enumerate(empty_traj_ids, start=1)
        ]

        retry_succeeded: list[str] = []
        retry_failed: list[str] = []
        for completed in asyncio.as_completed(retry_tasks):
            sample_id, error, stats = await completed
            if stats:
                test_stats[sample_id] = stats
            if error is None:
                retry_succeeded.append(sample_id)
            else:
                retry_failed.append(sample_id)

        print(f"补跑完成. success={len(retry_succeeded)} failed={len(retry_failed)}")
        if retry_failed:
            print("补跑失败样本:", retry_failed)

        # ── 重建 all_traj.jsonl，用补跑结果覆盖旧的空记录 ──────────────────
        # 读取当前文件，保留所有行；遇到同一 sample_id 时保留最后一条（补跑结果在后）
        ordered_ids: list[str] = []
        records: dict[str, dict] = {}
        if merged_jsonl_path.exists():
            with merged_jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sid = obj.get("sample_id", "")
                    if sid not in records:
                        ordered_ids.append(sid)
                    records[sid] = obj  # 后面的覆盖前面的

        with merged_jsonl_path.open("w", encoding="utf-8") as f:
            for sid in ordered_ids:
                f.write(json.dumps(records[sid], ensure_ascii=False) + "\n")
        print("all_traj.jsonl 已重建，空记录已被补跑结果覆盖。")
    else:
        print("\n无 trajectory 为空的样本，无需补跑。")

    # ── 汇总 pytest 测试统计 ────────────────────────────────────────────────
    if test_stats:
        total_passed = sum(s["passed"] for s in test_stats.values())
        total_failed = sum(s["failed"] for s in test_stats.values())
        total_tests = total_passed + total_failed
        all_pass_samples = [
            sid for sid, s in test_stats.items() if s["failed"] == 0 and s["total"] > 0
        ]
        any_fail_samples = [sid for sid, s in test_stats.items() if s["failed"] > 0]
        score = round(total_passed / total_tests * 100, 1) if total_tests > 0 else 0.0
        print(f"\n{'=' * 60}")
        print(f"测试汇总  ({len(test_stats)} 个样本)")
        print(
            f"  总用例: {total_tests}  passed: {total_passed}  failed: {total_failed}  score: {score}%"
        )
        print(f"  全部通过: {len(all_pass_samples)} 个样本")
        print(f"  存在失败: {len(any_fail_samples)} 个样本")
        if any_fail_samples:
            print(f"  失败样本: {sorted(any_fail_samples)}")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
