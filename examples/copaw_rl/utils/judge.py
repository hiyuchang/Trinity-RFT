from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

import yaml


GraderRef = Union[str, Callable[..., Any]]


@dataclass(frozen=True)
class TaskInfo:
    task_id: str
    prefix: str
    domain: str
    has_answer: bool
    task_path: Optional[Path]


# 1) 前缀归一化：把 task_id 的首段（_ 前）映射为统一前缀
# 2) 领域映射：用于日志/排查，可按需调整
PREFIX_ALIASES: Dict[str, str] = {
    "search": "search",
    "honey": "honey",
    "mat": "mat",
    "mm-tool": "mm-tool",
    "screen": "screen",
    "safety": "safety",
    "fr": "fr",
    "docx": "docx",
    "gov": "gov",
    "pdf": "pdf",
    "xlsx": "xlsx",
    "qa": "qa",
    "chinese_qa": "chinese_qa",
    "bootstrap": "bootstrap",
    "boostrap": "bootstrap",  # 兼容常见拼写
    "cron": "cron",
    "memory": "memory",
    "nl2bash": "nl2bash",
    "skill": "skill",
}

PREFIX_DOMAIN: Dict[str, str] = {
    "search": "search",
    "honey": "multimodal",
    "mat": "multimodal",
    "mm-tool": "multimodal+search",
    "screen": "screenshot",
    "safety": "safety",
    "fr": "fileprocess",
    "docx": "fileprocess",
    "gov": "fileprocess",
    "pdf": "fileprocess",
    "xlsx": "fileprocess",
    "qa": "qa",
    "chinese_qa": "qa",
    "bootstrap": "bootstrap",
    "cron": "cron",
    "memory": "memory",
    "nl2bash": "nl2bash",
    "skill": "skill",
    "sp": "systemprompt"
}

# grader 映射（domain -> grader）：
# - key: domain（先由 prefix 映射得到）
# - value.with_answer: evaluation.inputs.answer 非空时使用
# - value.without_answer: evaluation.inputs.answer 为空时使用
DOMAIN_GRADER_REGISTRY: Dict[str, Dict[str, GraderRef]] = {
    "search": {
        "with_answer": "graders.search:judge_grader_with_answer",
        "without_answer": "graders.search:judge_grader_without_answer",
    },
    "multimodal": {
        "with_answer": "graders.multimodal:judge_grader_with_answer",
        "without_answer": "graders.multimodal:judge_grader_without_answer",
    },
    "mm-tool": {
        "with_answer": "graders.mm_tool:judge_grader_with_answer",
        "without_answer": "graders.mm_tool:judge_grader_without_answer",
    },
    "screen": {
        "with_answer": "graders.gui:judge_grader_with_answer",
        "without_answer": "graders.gui:judge_grader_without_answer",
    },
    "safety": {
        "with_answer": "graders.safety:judge_grader_with_answer",
        "without_answer": "graders.safety:judge_grader_without_answer",
    },
    "fileprocess": {
        "with_answer": "graders.fileprocess:judge_grader_with_answer",
        "without_answer": "graders.fileprocess:judge_grader_without_answer",
    },
    "qa": {
        "with_answer": "graders.qa:judge_grader_with_answer",
        "without_answer": "graders.qa:judge_grader_without_answer",
    },
    "bootstrap": {
        "with_answer": "graders.bootstrap:judge_grader_with_answer",
        "without_answer": "graders.bootstrap:judge_grader_without_answer",
    },
    "cron": {
        "with_answer": "graders.cron:judge_grader_with_answer",
        "without_answer": "graders.cron:judge_grader_without_answer",
    },
    "memory": {
        "with_answer": "graders.memory:judge_grader_with_answer",
        "without_answer": "graders.memory:judge_grader_without_answer",
    },
    "nl2bash": {
        "with_answer": "graders.nl2bash:judge_grader_with_answer",
        "without_answer": "graders.nl2bash:judge_grader_without_answer",
    },
    "skill": {
        "with_answer": "graders.skill:judge_grader_with_answer",
        "without_answer": "graders.skill:judge_grader_without_answer",
    },
    "systemprompt": {
        "with_answer": "graders.systemprompt:judge_grader_with_answer",
        "without_answer": "graders.systemprompt:judge_grader_without_answer",
    },
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"task.yaml 根节点必须是 mapping: {path}")
    return data


def _extract_prefix(task_id: str) -> str:
    task_id = (task_id or "").strip()
    if not task_id:
        raise ValueError("metadata.task_id 不能为空")
    raw_prefix = task_id.split("_", 1)[0].strip().lower()
    return PREFIX_ALIASES.get(raw_prefix, raw_prefix)


def _answer_is_non_empty(raw_answer: Any) -> bool:
    if raw_answer is None:
        return False
    if isinstance(raw_answer, str):
        return bool(raw_answer.strip())
    if isinstance(raw_answer, (list, dict, tuple, set)):
        return len(raw_answer) > 0
    return True


def _extract_has_answer_from_session(session: Mapping[str, Any]) -> bool:
    """从 session 中提取最后一轮 response，判断是否为空。"""
    trajectory = session.get("agent", {}).get("_model_trajectory", [])
    if not isinstance(trajectory, list) or not trajectory:
        return False
    last = trajectory[-1]
    if not isinstance(last, Mapping):
        return False
    response = last.get("response")
    if isinstance(response, str):
        return _answer_is_non_empty(response)
    if isinstance(response, list):
        text_parts = []
        for item in response:
            if isinstance(item, Mapping):
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                text_parts.append(item)
        return _answer_is_non_empty("\n".join(text_parts))
    return _answer_is_non_empty(response)


def task_info_from_task_id(task_id: str, has_answer: bool) -> TaskInfo:
    """根据 task_id 和答案状态构造 TaskInfo（不依赖 task.yaml）。"""
    prefix = _extract_prefix(task_id)
    domain = PREFIX_DOMAIN.get(prefix, "unknown")
    return TaskInfo(
        task_id=task_id,
        prefix=prefix,
        domain=domain,
        has_answer=has_answer,
        task_path=None,
    )


def load_task_info(task_yaml_path: Union[str, Path]) -> tuple[TaskInfo, Dict[str, Any]]:
    task_path = Path(task_yaml_path).expanduser().resolve()
    data = _read_yaml(task_path)

    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata 必须是 mapping")

    evaluation = data.get("evaluation") or {}
    if not isinstance(evaluation, Mapping):
        raise ValueError("evaluation 必须是 mapping")
    inputs = evaluation.get("inputs") or {}
    if not isinstance(inputs, Mapping):
        raise ValueError("evaluation.inputs 必须是 mapping")

    task_id = str(metadata.get("task_id", "")).strip()
    prefix = _extract_prefix(task_id)
    domain = PREFIX_DOMAIN.get(prefix, "unknown")
    has_answer = _answer_is_non_empty(inputs.get("answer"))

    info = TaskInfo(
        task_id=task_id,
        prefix=prefix,
        domain=domain,
        has_answer=has_answer,
        task_path=task_path,
    )
    return info, data


def _resolve_grader(grader_ref: GraderRef) -> Callable[..., Any]:
    if callable(grader_ref):
        return grader_ref
    if not isinstance(grader_ref, str) or ":" not in grader_ref:
        raise ValueError(
            "grader 必须是可调用对象，或 'module.path:callable_name' 形式的字符串"
        )

    module_name, attr_name = grader_ref.split(":", 1)
    module = importlib.import_module(module_name)
    grader = getattr(module, attr_name, None)
    if not callable(grader):
        raise ValueError(f"未找到可调用 grader: {grader_ref}")
    return grader


def select_judge_grader(
    task_info: TaskInfo, registry: Dict[str, Dict[str, GraderRef]] | None = None
) -> Callable[..., Any]:
    registry = registry or DOMAIN_GRADER_REGISTRY
    by_domain = registry.get(task_info.domain)
    if not by_domain:
        raise KeyError(
            f"未找到 domain '{task_info.domain}' 的 grader 配置，请更新 DOMAIN_GRADER_REGISTRY"
        )

    key = "with_answer" if task_info.has_answer else "without_answer"
    grader_ref = by_domain.get(key)
    if grader_ref is None:
        raise KeyError(
            f"domain '{task_info.domain}' 缺少 key='{key}' 的 grader 配置"
        )

    return _resolve_grader(grader_ref)


def run_judge(task_yaml_path: Union[str, Path]) -> Any:
    task_info, task_data = load_task_info(task_yaml_path)
    grader = select_judge_grader(task_info)
    return grader(task_data=task_data, task_info=task_info)


def llm_judge(
    query: str,
    session: Mapping[str, Any],
    final_response: Any,
    task_id: str,
    input_answer: Any = None,
    **kwargs: Any,
) -> tuple[Any, str]:
    """供 run.py 调用：必须传 session/final_response；input_answer 非空时透传给 grader。"""
    has_answer = (
        _answer_is_non_empty(input_answer)
        if input_answer is not None
        else _extract_has_answer_from_session(session)
    )
    task_info = task_info_from_task_id(task_id=task_id, has_answer=has_answer)
    grader = select_judge_grader(task_info)

    # 必传字段：session / final_response；input_answer 非空时一并传入
    base_kwargs: Dict[str, Any] = {
        "query": query,
        "session": session,
        "final_response": final_response,
        "task_id": task_id,
        "task_info": task_info,
        **kwargs,
    }
    if _answer_is_non_empty(input_answer):
        base_kwargs["input_answer"] = input_answer

    # 兼容不同 grader 参数签名
    for attempt_kwargs in (
        base_kwargs,
        {k: v for k, v in base_kwargs.items() if k != "input_answer"},
        {
            "session": session,
            "final_response": final_response,
            "task_id": task_id,
            "query": query,
            **kwargs,
        },
        {"session": session, "final_response": final_response, "query": query, **kwargs},
        {"session": session, "final_response": final_response},
        {},
    ):
        try:
            result = grader(**attempt_kwargs)
            break
        except TypeError:
            continue
    else:
        raise TypeError(f"grader '{grader.__module__}.{grader.__name__}' 参数不匹配")

    if isinstance(result, tuple) and len(result) == 2:
        return result[0], str(result[1])
    return result, ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="根据 metadata.task_id 前缀和 evaluation.inputs.answer 是否为空选择 grader"
    )
    parser.add_argument(
        "task_yaml",
        nargs="?",
        default="task.yaml",
        help="task.yaml 路径（默认：当前目录 task.yaml）",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="仅输出选择到的 grader，不实际执行",
    )
    args = parser.parse_args()

    task_info, _ = load_task_info(args.task_yaml)
    grader = select_judge_grader(task_info)
    grader_name = f"{grader.__module__}.{grader.__name__}"

    print(
        f"task_id={task_info.task_id}, prefix={task_info.prefix}, domain={task_info.domain}, "
        f"has_answer={task_info.has_answer}, grader={grader_name}"
    )

    if not args.select_only:
        result = run_judge(args.task_yaml)
        print(f"judge_result={result}")


if __name__ == "__main__":
    main()
