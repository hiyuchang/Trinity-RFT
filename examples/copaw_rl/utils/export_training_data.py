import json
import os
import logging
import sys

import numpy as np
from urllib import request as urllib_request


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# 脚本自身所在目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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


def export_training_data(task_id, trajectories) -> None:
    # ── LLM 判断是否执行成功 ──────────────────────────────────────────
    try:
        judge_ok, judge_reason = _llm_judge(task_id, trajectories)
    except Exception as judge_exc:
        # 判断出错时保守处理：视为成功，避免误丢弃
        judge_ok, judge_reason = True, f"LLM判断异常(视为成功): {judge_exc}"
    log.info("LLM judge result: %s, reason: %s", judge_ok, judge_reason)

    dataset = []
    last_full_token_ids = last_full_length = None
    for trajectory in trajectories:
        logprobs = trajectory["logprobs"]
        prompt_token_ids = trajectory["prompt_token_ids"]
        token_ids = trajectory["token_ids"]
        assert len(logprobs) == len(
            token_ids
        ), f"logprobs和token_ids长度不匹配, {len(logprobs)=} {len(token_ids)=}"

        np_prompt_token_ids = np.array(prompt_token_ids)
        if last_full_token_ids is not None and np.array_equal(
            last_full_token_ids, np_prompt_token_ids[:last_full_length]
        ):
            log.info("与上一条完全匹配，合并数据")
            last_data = dataset[-1]
            pad_len = len(np_prompt_token_ids) - last_full_length
            last_data["token_ids"] += prompt_token_ids[last_full_length:] + token_ids
            last_data["logprobs"] += [0.0] * pad_len + logprobs
            last_data["response_mask"] += [0] * pad_len + [1] * len(token_ids)
        else:
            if last_full_token_ids is not None and last_full_length <= len(np_prompt_token_ids):
                mismatch_position = np.where(
                    (last_full_token_ids != np_prompt_token_ids[:last_full_length])
                )[0][0]
                log.info(
                    f"添加新数据, {mismatch_position.item() = }, "
                    f"{last_full_token_ids[mismatch_position:mismatch_position+10].tolist()} vs "
                    f"{np_prompt_token_ids[mismatch_position:mismatch_position+10].tolist()}"
                )
            data = {
                "logprobs": logprobs,
                "prompt_token_ids": prompt_token_ids,
                "token_ids": token_ids,
                "response_mask": [1] * len(token_ids),
                "judge_ok": judge_ok,
            }
            dataset.append(data)

        last_full_token_ids = np.array(prompt_token_ids + token_ids)
        last_full_length = len(last_full_token_ids)

    with open(os.path.join(_SCRIPT_DIR, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)  # , indent=2
