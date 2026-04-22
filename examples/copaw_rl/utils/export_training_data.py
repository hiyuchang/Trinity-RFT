import json
import os
import logging
import sys

import numpy as np
from urllib import request as urllib_request
from judge import llm_judge as dispatch_llm_judge

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


def _llm_judge(
    query: str,
    session_data: dict,
    final_response: str,
    task_id: str,
    input_answer,
) -> tuple[float, str]:
    """通过 utils/judge.py 分发到对应 grader。"""
    return dispatch_llm_judge(
        query=query,
        session=session_data,
        final_response=final_response,
        task_id=task_id,
        input_answer=input_answer,
    )


def _extract_text(content) -> str:
    """尽量把 message/response 的 content 抽成纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
    return str(content)


def _extract_first_user_query(trajectories: list) -> str:
    """从 trajectories 中提取首条 user 消息文本。"""
    for entry in trajectories:
        if not isinstance(entry, dict):
            continue
        for msg in entry.get("messages", []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                text = _extract_text(msg.get("content"))
                if text:
                    return text
    return ""


def _extract_final_response(trajectories: list):
    """提取最后一轮 response 的文本；没有时返回空字符串。"""
    if not trajectories:
        return ""
    last = trajectories[-1]
    if not isinstance(last, dict):
        return ""
    response = last.get("response", "")
    if isinstance(response, str):
        return response
    return _extract_text(response)


def export_training_data(task_id, trajectories, session_data=None) -> None:
    try:
        query = _extract_first_user_query(trajectories)
        final_response = _extract_final_response(trajectories)
        judge_session = (
            session_data
            if isinstance(session_data, dict)
            else {"agent": {"_model_trajectory": trajectories}}
        )
        reward, judge_reason = _llm_judge(
            query=query,
            session_data=judge_session,
            final_response=final_response,
            task_id=task_id,
            input_answer=None,
        )
    except Exception as judge_exc:
        # 判断出错时保守处理：视为失败
        reward, judge_reason = 0.0, f"LLM判断异常(失败): {judge_exc}"
    log.info("LLM judge result: %s, reason: %s", reward, judge_reason)

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
                "reward": reward,
            }
            dataset.append(data)

        last_full_token_ids = np.array(prompt_token_ids + token_ids)
        last_full_length = len(last_full_token_ids)

    with open(os.path.join(_SCRIPT_DIR, "dataset.json"), "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)  # , indent=2
