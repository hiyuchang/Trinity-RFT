#!/usr/bin/env python3
"""
batch_run.py

批量运行 benchmark 任务。每个任务独立创建一个容器环境。

用法:
  # 跑全部未注释的任务（使用默认模型）
  python batch_run.py

  # 跑指定任务（推荐用 --tasks 避免任务名被误解析为模型）
  python batch_run_v2.py --models kimi-k2.5 --tasks mat_0001_en
  python batch_run_v2.py 097-bootstrap-modify-identity-xiaolu --models MiniMax-M2.5  # 或把任务放前面

  # 跑指定 package（按分类目录跑所有子任务）
  python batch_run.py --package cron
  python batch_run.py --package cron dingtalk
  python batch_run.py --package file_processing --parallel 4
  python batch_run.py --package security
  python batch_run.py --package search security --parallel 8

  # 跑指定范围
  python batch_run.py --range 1 8

  # 并行跑（默认 4 个并发）
  python batch_run.py --parallel 4

  # 串行跑（等同于 --parallel 1）
  python batch_run.py --serial

  # 多模型 × 多任务并行跑
  python batch_run.py --models qwen3.5-plus qwen-max

  # 指定多个模型并发跑全部任务
  python batch_run.py --models qwen3.5-plus --parallel 4

  # 从 JSON 文件加载模型配置
  python batch_run.py --models-file models.json

  # 跑完后自动重试 ERROR 任务（最多 2 轮）
  python batch_run_v2.py --models-file models.json --package cron --retry-errors 2

  # 每个模型重复推理 3 次，计算平均分
  python batch_run_v2.py --models-file models.json --trial 3 --package cron
  # 也可在 JSON 配置中为每个模型单独设置 trial 字段（--trial 全局覆盖）

  # 从已有结果目录重跑 ERROR 任务
  python batch_run_v2.py --retry-from /mnt/xielipeng.xlp/new/CoPaw-Pro/result/20260330/20260330_213935 --retry-errors 3
  python batch_run_v2.py --retry-from result/20260330/20260330_213935 --models-file eval_0320_gguf.json --retry-errors 3

  # 运行日志自动写入 logs/{YYYYMMDD}/batch_run_{HHMMSS}.log（与 auto_eval 一致）；仍可 nohup 重定向一份
  nohup env PYTHONUNBUFFERED=1 python batch_run_v2.py --models glm-5 --package file_processing search QA gov bootstrap memory cron other --parallel 16 &
  
  nohup env PYTHONUNBUFFERED=1 python batch_run_v2.py --models glm-5 MiniMax-M2.5 kimi-k2.5 qwen3.5-plus gpt-5.4 qwen3.5-397b-a17b --package file_processing search QA gov bootstrap memory cron other --parallel 16 &

nohup env PYTHONUNBUFFERED=1 python batch_run_v2.py --models qwen3.5-plus --package file_processing search QA gov bootstrap memory cron other multimodel_search skill safety --parallel 16 &

  nohup env PYTHONUNBUFFERED=1 python batch_run_v2.py --models qwen3.5-plus --package corn guide_assistance --parallel 8 &

  nohup env PYTHONUNBUFFERED=1 python batch_run_v2.py --models-file eval_flash_model.json --package safety --parallel 8 &

  nohup env PYTHONUNBUFFERED=1 python batch_run_v2.py --models-file eval_pai_models.json --package file_processing search QA gov bootstrap memory cron other claweval --parallel 16 &

  vllm serve /path/to/hf --dtype bfloat16 --enable-auto-tool-choice --tool-call-parser qwen3_xml --tensor-parallel-size 1 --data-parallel-size 8
"""

import argparse
import asyncio
import base64
import copy
import json
import os
import random
import re
import sys
import time
import traceback
import zipfile
from datetime import datetime
import importlib

spec = importlib.util.spec_from_file_location(
    "sandbox_utils",
    os.path.join(os.path.dirname(__file__), "..", "workflows", "sandbox_utils.py"),
)
sandbox_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sandbox_utils)
get_or_create_sandbox = sandbox_utils.get_or_create_sandbox
run_eval_workflow = sandbox_utils.run_eval_workflow


class _TeeIO:
    """同时将输出写到多个流（控制台 + 日志文件）。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            try:
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return self.streams[0].isatty() if self.streams else False


class _BatchStdoutTee:
    """batch_run 主流程：stdout/stderr 双写到 logs/{date}/batch_run_{HHMMSS}.log。"""

    def __init__(self):
        self._orig_out = None
        self._orig_err = None
        self._fp = None

    def start(self) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_base = os.path.join(script_dir, "logs")
        date_str = os.environ.get("RESULT_DATE_PREFIX", datetime.now().strftime("%Y%m%d"))
        ts = datetime.now().strftime("%H%M%S")
        day_dir = os.path.join(log_base, date_str)
        os.makedirs(day_dir, exist_ok=True)
        log_path = os.path.join(day_dir, f"batch_run_{ts}.log")
        self._fp = open(log_path, "w", encoding="utf-8")
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        sys.stdout = _TeeIO(self._orig_out, self._fp)
        sys.stderr = _TeeIO(self._orig_err, self._fp)
        return log_path

    def stop(self):
        if self._orig_out is not None:
            sys.stdout = self._orig_out
        if self._orig_err is not None:
            sys.stderr = self._orig_err
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        self._orig_out = self._orig_err = None


PACKAGE_TASKS: dict[str, list[str]] = {
    # "local_remote": [
    #     "local_bench_easy_1",
    #     "local_bench_hard_1",
    #     "local_bench_hard_2",
    # ],
    "guide_assistance": [
        "225-multi-config-skill-cron-agent",
        "226-news-anchor-skill-cron-agent",
        "227-weekly-report-skill-cron-agent",
        "228-market-analyst-skill-cron-agent",
        "229-scrum-master-skill-cron-agent",
        "230-study-buddy-skill-cron-agent",
        "231-tech-radar-skill-cron-agent",
        "232-sre-monitor-skill-cron-agent",
        "233-translator-skill-cron-agent",
        "234-doc-butler-skill-cron-agent",
        "235-data-analyst-skill-cron-agent",
    ],
    "multimodel_search": [
        "mm_tool_7990",
        "mm_tool_7991",
        "mm_tool_7992",
        "mm_tool_7993",
        "mm_tool_7994",
        "mm_tool_7995",
        "mm_tool_7997",
        "mm_tool_7998",
        "mm_tool_8000",
        "mm_tool_aug_001",
        "mm_tool_aug_002",
        "mm_tool_aug_003",
        "mm_tool_aug_004",
        "mm_tool_aug_005",
        "mm_tool_aug_007",
        "mm_tool_aug_008",
        "mm_tool_aug_011",
        "mm_tool_aug_012",
        "mm_tool_aug_013",
        "mm_tool_aug_014",
        "mm_tool_aug_015",
        "mm_tool_aug_016",
        "mm_tool_aug_017",
        "mm_tool_aug_018",
        "mm_tool_aug_022",
        "mm_tool_aug_024",
        "mm_tool_aug_026",
        "mm_tool_aug_030",
        "mm_tool_aug_034",
        "mm_tool_aug_035",
        "mm_tool_aug_036",
        "mm_tool_aug_037",
        "mm_tool_aug_038",
    ],
    "skill": [
        "skillhub_0055",
        "skillhub_0100",
        "skillhub_0128",
        "skillhub_0141",
        "skillhub_0172",
        "skillhub_0189",
        "skillhub_0211",
        "skillhub_0212",
        "skillhub_0249",
        "skillhub_0256",
        "skillhub_0259",
        "skillhub_0265",
        "skillhub_0290",
        "skillhub_0296",
        "skillbench_001",
        "skillbench_002",
        "skillbench_003",
        "skillbench_007",
        "skillbench_008",
        "skillbench_009",
        "skillbench_010"
    ],
    "safety": [
        "safety_001_zh",
        "safety_002_zh",
        "safety_003_zh",
        "safety_004_zh",
        "safety_005_zh",
        "safety_006_zh",
        "safety_007_zh",
        "safety_008_zh",
        "safety_009_zh",
        "safety_010_zh",
        "safety_da_conf_disc_0001_zh",
        "safety_da_cont_inje_0001_zh",
        "safety_da_cred_expo_0001_zh",
        "safety_ha_cybe_atta_faci_0001_zh",
        "safety_ha_dang_syst_comm_0001_zh",
        "safety_ha_fina_loss_0001_zh",
        "safety_ha_mali_code_inje_0001_zh",
        "safety_ha_mali_cont_requ_0001_zh",
        "safety_ha_priv_leak_0001_zh",
        "safety_ha_self_harm_faci_0001_zh",
        "safety_pr_emot_pres_0001_zh",
        "safety_pr_enco_bypa_0001_zh",
        "safety_pr_iden_spoo_0001_zh",
        "safety_pr_incr_prob_0001_zh",
        "safety_sa_norm_code_task_0001_zh",
        "safety_sa_norm_file_oper_0001_zh",
        "safety_sa_norm_skil_usag_0001_zh",
    ],
    "QA": [
        "034-copaw-change-model",
        "035-copaw-configure-dashscope",
        "036-copaw-add-env",
        "037-copaw-configure-channel",
        "038-copaw-feishu-issue",
        "039-copaw-add-skill",
        "040-copaw-add-mcp",
        "041-copaw-check-mail",
        "042-copaw-check-update",
        "195-copaw-create-cron",
        "196-copaw-magic-commands",
        "197-copaw-tool-guard",
        "198-copaw-memory-usage",
        "199-copaw-pdf-skill",
        "200-copaw-heartbeat",
        "201-copaw-daemon-status",
    ],
    "bootstrap": [
        "097-bootstrap-modify-identity-xiaolu",
        "098-bootstrap-update-user-xiaolin",
        "099-bootstrap-modify-identity-xiaolu2",
        "100-bootstrap-update-user-xiaofei",
        "101-bootstrap-modify-identity-xiaonuan",
        "102-bootstrap-update-user-note",
        "103-bootstrap-modify-identity-azhe",
        "104-bootstrap-update-user-xiaolin2",
        "161-bootstrap-modify-identity-xiaoxing",
        "162-bootstrap-update-user-ajie",
        "163-bootstrap-modify-identity-xiaoyue",
        "164-bootstrap-update-user-xiaoyu",
        "165-bootstrap-modify-identity-laok",
        "166-bootstrap-update-user-dazhuang",
        "167-bootstrap-modify-identity-xiaoqi",
        "168-bootstrap-update-user-xiaomi",
        "169-bootstrap-modify-identity-doudou",
    ],
    "cron": [
        "007-daily-water-reminder",
        "008-multiple-reminders",
        "009-us-stock-market-reminder",
        "010-arxiv-daily-summary",
        "013-agent-rl-paper-scheduler",
        "069-cron-astock-report",
        "070-cron-delete-weather",
        "071-cron-multi-reminders",
        "170-cron-bedtime-reading",
        "171-cron-weekly-report",
        "172-cron-delete-water",
        "173-cron-workday-news",
        "174-cron-fitness-reminders",
        "175-cron-birthday-annual",
        "176-cron-replace-meeting",
        "177-cron-weekend-cleanup",
    ],
    "file_processing": [
        "017-word",
        "018-xlsx-experiment",
        "019-pdf-invoice-extract",
        "020-pdf-paper-read",
        "021-file-copaw-install",
        "022-file-experiment",
        "023-file-python",
        "024-list-files",
        "043-docx-extract-plan",
        "044-docx-compare-meetings",
        "045-docx-create-from-pdf",
        "046-pdf-extract-text",
        "047-pdf-search-rl",
        "048-pdf-extract-table",
        "049-pdf-read-summary",
        "050-pdf-compare-invoices",
        "051-pdf-to-xlsx",
        "052-pdf-to-docx",
        "053-pdf-extract-images-qwen",
        "054-pdf-extract-images-bert",
        "055-pdf-extract-text-minigpt4",
        "056-xlsx-search-latency",
        "057-xlsx-convert-tsv",
        "058-file-readme-section",
        "059-file-gpu-vendor",
        "060-file-log-cuda-oom",
        "061-file-log-compare",
        "064-file-code-explain",
        "065-file-csv-filter",
        "066-file-code-review",
        "067-file-code-summary",
        "068-file-log-rag-warning",
    ],
    "gov": [
        "114-html-extract-named-entities",
        "116-html-extract-abbreviations",
        "120-xml-extract-markland-dam-terms",
        "121-html-extract-hawkeye-rfi",
        "122-txt-simplify-job-posting",
        "123-html-extract-mqsa-statistics",
        "124-pdf-analyze-modification-rationale",
        "126-search-earthquake-related-docs",
        "127-html-design-wildflower-questions",
        "128-html-extract-year",
        "129-html-extract-deadlines",
        "131-csv-group-budget-by-agency",
        "132-csv-filter-wind-speed-60mph",
        "134-pdf-extract-references",
        "136-html-analyze-sec-form5a",
        "137-html-extract-document-publisher",
        "138-txt-extract-usgs-spectral-terms",
        "139-pdf-extract-first-author",
    ],
    "memory": [
        "003-output-preference",
        "004-memory-email",
        "072-memory-markdown-table",
        "073-memory-chinese-output",
        "074-memory-markdown-report",
        "075-memory-concise-reply",
        "076-memory-markdown-format",
        "077-memory-chinese-pref",
        "078-memory-save-markdown",
        "079-memory-markdown-table-tz",
        "080-memory-concise-style",
        "186-memory-english-output",
        "187-memory-code-comment-cn",
        "188-memory-detail-explain",
        "189-memory-step-by-step",
        "190-memory-no-emoji",
        "191-memory-formal-tone",
        "192-memory-timezone-sh",
        "193-memory-yaml-config",
        "194-memory-list-format",
    ],
    "claweval": [
        "T18-ticket-triage",
        "T29-cross-service-meeting"
    ],
    "multimodel": [
        "honey_00137_en",
        "honey_00173_en",
        "honey_00317_en",
        "honey_00322_zh",
        "honey_00332_en",
        "honey_00345_en",
        "honey_00345_zh",
        "honey_00504_en",
        "honey_00788_zh",
        "honey_01036_en",
        "honey_01036_zh",
        "honey_01295_zh",
        "honey_01547_zh",
        "honey_01693_en",
        "honey_01693_zh",
        "honey_01908_en",
        "honey_02359_en",
        "honey_02741_zh",
        "honey_02749_zh",
        "honey_03119_en",
        "honey_03130_en",
        "honey_03263_zh",
        "honey_03449_en",
        "honey_04283_zh",
        "honey_04343_zh",
        "honey_04812_zh",
        "honey_04891_en",
        "honey_04968_zh",
        "honey_05219_zh",
        "honey_05722_zh",
        "honey_05772_en",
        "honey_06031_zh",
        "honey_06742_zh",
        "honey_06785_zh",
        "honey_06988_en",
        "honey_06988_zh",
        "honey_07077_en",
        "honey_07141_zh",
        "honey_07350_en",
        "honey_07448_zh",
        "mat_0001_en",
        "mat_0001_zh",
        "mat_0002_en",
        "mat_0007_en",
        "mat_0010_en",
        "mat_0014_zh",
        "mat_0016_zh",
        "mat_0023_en",
        "mat_0023_zh",
        "mat_0024_zh",
        "mat_0025_en",
        "mat_0030_en",
        "mat_0033_zh",
        "mat_0036_en",
        "mat_0041_zh",
    ],
    "other": [
        "001-team-building-cities",
        "002-resume",
        "005-mcp-list",
    ],
    "gui": [
        "gui_001_zh",
        "gui_002_zh",
        "gui_003_zh",
        "gui_004_zh",
        "gui_005_zh",
        "gui_006_zh",
        "gui_007_zh",
        "gui_008_zh",
        "gui_009_zh",
        "gui_010_zh",
        "gui_011_zh",
        "gui_012_zh",
        "gui_013_zh",
        "gui_014_zh",
        "screen_da_admi_pane_0020_zh",
        "screen_da_anal_dash_0004_zh",
        "screen_da_sett_page_0009_zh",
        "screen_da_sett_page_0013_zh",
        "screen_do_wiki_page_0010_zh",
    ],
    "search": [
        "030-news-tech",
        "031-browser-arxiv-read",
        "032-news-summary",
        "033-browser-screenshot",
        "081-search-nextjs-router",
        "082-search-rag-papers",
        "083-search-llm-progress",
        "084-search-alibaba-metro",
        "085-search-taobao-phone",
        "086-search-apple-watch",
        "087-news-finance-summary",
        "088-search-docker-wsl",
        "178-news-sports",
        "179-browser-github-trending",
        "180-search-k8s-ingress",
        "181-browser-douban-movie",
        "182-news-international",
        "183-search-rust-async",
        "184-browser-hacker-news",
        "185-search-ev-ranking",
        # 难样本：多步检索 / browser / news（202-214）
        "202-search-travel-japan-plan",
        "203-search-phone-upgrade-cost",
        "204-browser-github-rag-tool",
        "205-search-finance-rate-compare",
        "206-search-rent-subsidy-policy",
        "207-browser-recipe-compare",
        "208-search-ev-car-compare",
        "209-search-xz-backdoor-timeline",
        "210-browser-so-memory-leak",
        "211-search-shanghai-concert",
        "212-news-ai-regulation-impact",
        "213-news-real-estate-trend",
        "214-news-tech-company-moves",
    ],
    "document_understanding": [
        "M018_doc_extraction_line_chart",
        "M019_doc_extraction_radar_chart",
        "M020_multi_doc_extraction_bar_chart",
        "M073_doc_extraction_training_cost",
        "M074_doc_extraction_thinking_impact",
        "M076_doc_extraction_cross_table_merge",
        "M079_doc_extraction_f1_verification",
        "M080_doc_extraction_delta_comparison",
        "M081_doc_extraction_heatmap_comparison",
        "M084_doc_figure_reproduction_pie",
        "T077_officeqa_highest_dept_spending",
        "T078_officeqa_max_yield_spread",
        "T080_officeqa_bond_yield_change",
        "T081_officeqa_cagr_trust_fund",
        "T085_officeqa_army_expenditures",
        "T098_pinbench_openclaw_facts",
    ],
}

PACKAGE_CATEGORIES = sorted(PACKAGE_TASKS.keys())

PROVIDER_BASE = {
    "providers": {
        "modelscope": {
            "base_url": "https://api-inference.modelscope.cn/v1",
            "api_key": "",
            "extra_models": [],
            "chat_model": ""
        },
        "dashscope": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
            "extra_models": [],
            "chat_model": ""
        },
        "aliyun-codingplan": {
            "base_url": "https://coding.dashscope.aliyuncs.com/v1",
            "api_key": "",
            "extra_models": [],
            "chat_model": ""
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "",
            "extra_models": [],
            "chat_model": ""
        },
        "azure-openai": {
            "base_url": "",
            "api_key": "",
            "extra_models": [],
            "chat_model": ""
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
            "extra_models": [],
            "chat_model": ""
        }
    },
    "custom_providers": {
        "local-4b": {
            "id": "local-4b",
            "name": "local-4b",
            "default_base_url": "",
            "api_key_prefix": "",
            "models": [
                {
                    "id": "/nas/checkpoints/Qwen3.5-4B",
                    "name": "4b"
                }
            ],
            "base_url": "http://101.37.165.227:8081/v1",
            "api_key": "",
            "chat_model": "OpenAIChatModel"
        }
    },
    "active_llm": {
        "provider_id": "dashscope",
        "model": "qwen3.5-plus"
    }
}


MODELS = {
    "MiniMax-M2.5": {"provider_id": "dashscope", "model": "MiniMax-M2.5"},
    "glm-5": {"provider_id": "dashscope", "model": "glm-5"},
    "kimi-k2.5": {"provider_id": "dashscope", "model": "kimi-k2.5"},
    "qwen3.5-plus": {"provider_id": "dashscope", "model": "qwen3.5-plus"},
    "qwen3.6-plus": {"provider_id": "dashscope", "model": "qwen3.6-plus"},
    "gpt-5.4": {"provider_id": "dashscope", "model": "openai.gpt-5.4-2026-03-05"},
    "claude-opus-4-6": {"provider_id": "dashscope", "model": "vertex_ai.claude-opus-4-6"},
    "gemini-3.1-pro": {"provider_id": "dashscope", "model": "vertex_ai.gemini-3.1-pro-preview"},
    "grok-4-1-fast-reasoning": {"provider_id": "dashscope", "model": "grok-4-1-fast-reasoning"},
    "qwen3.5-397b-a17b": {"provider_id": "dashscope", "model": "qwen3.5-397b-a17b"}
}

DEFAULT_MODEL = "qwen3.5-plus"


def build_provider_config(model_key: str) -> dict:
    """基于 PROVIDER_BASE 模板，切换 active_llm 生成对应模型的 provider 配置。

    支持环境变量 AUTO_EVAL_BASE_URL / AUTO_EVAL_MODEL_ID 动态注入 provider，
    供 auto_eval.py 自动化流水线使用。
    """
    auto_base_url = os.environ.get("AUTO_EVAL_BASE_URL")
    auto_model_id = os.environ.get("AUTO_EVAL_MODEL_ID")

    if auto_base_url and auto_model_id and model_key == os.environ.get("AUTO_EVAL_MODEL_KEY"):
        cfg = copy.deepcopy(PROVIDER_BASE)
        provider_id = f"auto-{model_key}"
        custom_provider = {
            "id": provider_id,
            "name": provider_id,
            "default_base_url": "",
            "api_key_prefix": "",
            "models": [{"id": auto_model_id, "name": model_key, "supports_multimodal": True, "supports_image": True, "supports_video": False}],
            "base_url": auto_base_url,
            "api_key": "",
            "chat_model": "OpenAIChatModel",
        }
        generate_kwargs_str = os.environ.get("AUTO_EVAL_GENERATE_KWARGS")
        if generate_kwargs_str:
            try:
                custom_provider["generate_kwargs"] = json.loads(generate_kwargs_str)
            except json.JSONDecodeError:
                pass
        cfg["custom_providers"][provider_id] = custom_provider
        cfg["active_llm"] = {"provider_id": provider_id, "model": auto_model_id}
        return cfg

    if model_key in MODELS:
        llm_config = MODELS[model_key]
    else:
        if ":" in model_key:
            provider_id, model_name = model_key.split(":", 1)
            llm_config = {"provider_id": provider_id, "model": model_name}
        else:
            llm_config = {"provider_id": "dashscope", "model": model_key}
    cfg = copy.deepcopy(PROVIDER_BASE)
    cfg["active_llm"] = llm_config

    # Ensure the target model is in the provider's extra_models so CoPaw can find it
    pid = llm_config.get("provider_id", "")
    mid = llm_config.get("model", "")
    if pid and mid and pid in cfg.get("providers", {}):
        provider_cfg = cfg["providers"][pid]
        existing = provider_cfg.get("extra_models", [])
        if not any(m.get("id") == mid for m in existing):
            provider_cfg["extra_models"] = [*existing, {
                "id": mid, "name": mid,
                "supports_multimodal": True,
                "supports_image": True,
                "supports_video": False,
            }]

    return cfg

def build_provider_config_from_json(item: dict) -> dict:
    """从 JSON 配置项构建 provider config。

    支持两种格式:
    1. 新格式 (推荐): {"key": "名称", "base_url": "...", "model_id": "...", "api_key": ""}
    2. 旧格式: {"key": "名称", "provider_id": "dashscope", "model": "qwen3.5-plus"}
    """
    key = item["key"]

    if "base_url" in item:
        cfg = copy.deepcopy(PROVIDER_BASE)
        provider_id = f"json-{key}"
        model_id = item.get("model_id", key)
        cfg["custom_providers"][provider_id] = {
            "id": provider_id,
            "name": key,
            "default_base_url": "",
            "api_key_prefix": "",
            "models": [{"id": model_id, "name": key, "supports_multimodal": True, "supports_image": True, "supports_video": False}],
            "base_url": item["base_url"],
            "api_key": item.get("api_key", ""),
            "chat_model": "OpenAIChatModel",
        }
        cfg["active_llm"] = {"provider_id": provider_id, "model": model_id}
        return cfg

    if "provider_id" in item and "model" in item:
        cfg = build_provider_config(key)
        cfg["active_llm"] = {
            "provider_id": item["provider_id"],
            "model": item["model"],
        }
        return cfg

    return build_provider_config(key)


def get_tasks_from_packages(packages: list[str]) -> list[str]:
    """Return task IDs for the given package names from PACKAGE_TASKS."""
    task_ids = []
    for pkg in packages:
        if pkg not in PACKAGE_TASKS:
            print(f"[WARN] Unknown package: {pkg}. Available: {', '.join(PACKAGE_CATEGORIES)}")
            continue
        task_ids.extend(PACKAGE_TASKS[pkg])
    return task_ids


ALL_TASKS = sorted(set(t for tasks in PACKAGE_TASKS.values() for t in tasks))

TASK_TO_CATEGORY: dict[str, str] = {}
for _cat, _tasks in PACKAGE_TASKS.items():
    for _t in _tasks:
        TASK_TO_CATEGORY[_t] = _cat


def load_error_tasks_from_dir(result_dir: str) -> tuple[dict, list[tuple[str, str]]]:
    """从结果目录的 _batch_summary.json 读取 ERROR 状态的任务。

    Returns:
        (summary_dict, [(model_label, task_id), ...])
    """
    summary_path = os.path.join(result_dir, "_batch_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"找不到 {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    for r in summary.get("results", []):
        if r["status"] != "ERROR" and r.get("score", -1) <= 0 and r.get("steps", -1) <= 0:
            r["status"] = "ERROR"

    error_pairs = [
        (r.get("model", ""), r["task"])
        for r in summary.get("results", [])
        if r["status"] == "ERROR"
    ]
    return summary, error_pairs


def infer_trial_groups(model_keys: list[str]) -> dict[str, list[str]]:
    """从模型 key 列表中推断 trial 分组（检测 _tN 后缀模式）。

    例如 ["pai-8b_t1", "pai-8b_t2", "pai-8b_t3", "other"]
      → {"pai-8b": ["pai-8b_t1", "pai-8b_t2", "pai-8b_t3"]}
    """
    groups: dict[str, list[str]] = {}
    for key in model_keys:
        m = re.match(r'^(.+)_t(\d+)$', key)
        if m:
            groups.setdefault(m.group(1), []).append(key)
    return {base: sorted(keys) for base, keys in groups.items() if len(keys) > 1}


def merge_retry_results(original_results: list[dict], retry_results: list[dict]) -> list[dict]:
    """将重试结果合并到原始结果中，替换对应的条目。"""
    retry_map = {(r.get("model", ""), r["task"]): r for r in retry_results}
    return [
        retry_map.get((r.get("model", ""), r["task"]), r)
        for r in original_results
    ]


def save_batch_summary(batch_dir: str, all_results: list[dict],
                       model_keys: list[str], task_ids: list[str],
                       max_parallel: int, run_ts: str,
                       multi_model: bool) -> str:
    """构建 batch summary 并保存到 _batch_summary.json，返回文件路径。"""
    os.makedirs(batch_dir, exist_ok=True)
    total_passed = sum(1 for r in all_results if r["status"] == "PASS")
    total_failed = sum(1 for r in all_results if r["status"] == "FAIL")
    total_errors = sum(1 for r in all_results if r["status"] == "ERROR")
    total_evaluated = total_passed + total_failed
    category_stats = compute_category_stats(all_results)
    batch_summary = {
        "timestamp": run_ts,
        "models": model_keys,
        "parallel": max_parallel,
        "total": len(all_results),
        "evaluated": total_evaluated,
        "passed": total_passed,
        "failed": total_failed,
        "errors": total_errors,
        "category_stats": category_stats,
        "results": all_results,
    }
    if multi_model:
        score_map: dict[str, dict] = {}
        steps_map: dict[str, dict] = {}
        length_map: dict[str, dict] = {}
        duration_map: dict[str, dict] = {}
        status_map: dict[str, dict] = {}
        for r in all_results:
            m = r.get("model", "")
            score_map.setdefault(m, {})[r["task"]] = r["score"]
            steps_map.setdefault(m, {})[r["task"]] = r.get("steps", -1)
            length_map.setdefault(m, {})[r["task"]] = r.get("response_length", -1)
            duration_map.setdefault(m, {})[r["task"]] = r.get("duration_seconds", -1)
            status_map.setdefault(m, {})[r["task"]] = r["status"]
        model_stats = {}
        for m in model_keys:
            scores = score_map.get(m, {})
            statuses = status_map.get(m, {})
            valid_scores = [s for t, s in scores.items() if statuses.get(t) != "ERROR" and s >= 0]
            steps = steps_map.get(m, {})
            valid_steps = [s for t, s in steps.items() if statuses.get(t) != "ERROR" and s >= 0]
            lengths = length_map.get(m, {})
            valid_lengths = [s for t, s in lengths.items() if statuses.get(t) != "ERROR" and s >= 0]
            durations = duration_map.get(m, {})
            valid_durations = [s for t, s in durations.items() if statuses.get(t) != "ERROR" and s >= 0]
            n_errors = sum(1 for s in statuses.values() if s == "ERROR")
            model_stats[m] = {
                "avg_score": sum(valid_scores) / len(valid_scores) if valid_scores else -1,
                "avg_steps": sum(valid_steps) / len(valid_steps) if valid_steps else -1,
                "avg_response_length": int(sum(valid_lengths) / len(valid_lengths)) if valid_lengths else -1,
                "avg_duration_seconds": sum(valid_durations) / len(valid_durations) if valid_durations else -1,
                "passed": sum(1 for s in valid_scores if s == 100),
                "total": len(task_ids),
                "evaluated": len(task_ids) - n_errors,
                "errors": n_errors,
                "scores": scores,
                "steps": steps,
                "response_lengths": lengths,
                "duration_seconds": durations,
            }
        batch_summary["model_stats"] = model_stats
    summary_path = os.path.join(batch_dir, "_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总已保存到: {summary_path}")
    return summary_path


def _is_infra_error(e: Exception) -> bool:
    """判断是否为基础设施瞬态错误（可重试）。"""
    if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
        return True
    if isinstance(e, RuntimeError):
        msg = str(e)
        if "Failed to create environment" in msg:
            return True
    return False


async def run_single_task(
    sample_id: str,
    run_ts: str,
    provider_config: dict | None = None,
    model_label: str = "",
    semaphore: asyncio.Semaphore | None = None,
    result_root: str = "result",
    max_retries: int = 3
) -> dict:
    """运行单个任务，返回结果摘要。支持信号量控制并发。"""
    # if provider_config is None:
    #     provider_config = build_provider_config(DEFAULT_MODEL)
    tag = f"[{model_label}] {sample_id}" if model_label else sample_id

    def _inner():
        from trinity.utils.log import get_logger
        logger = get_logger()
        token = os.environ.get("E2B_TOKEN")
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
        api_server_url = os.environ.get("AUTO_EVAL_BASE_URL")
        model_path = os.environ.get("AUTO_EVAL_MODEL_ID")

        try:
            metrics = run_eval_workflow(
                sandbox,
                sample_id,
                oss_config,
                dashscope_api_key,
                api_server_url,
                model_path,
                model_label,
                os.path.join(result_root, run_ts),
                logger,
            )
        finally:
            sandbox.kill()
        return metrics

    for attempt in range(max_retries + 1):
        try:
            if semaphore:
                async with semaphore:
                    return await asyncio.to_thread(_inner)
            return await asyncio.to_thread(_inner)
        except Exception as e:
            if attempt < max_retries and _is_infra_error(e):
                wait = min(2 ** attempt * 5, 60)
                print(f"[RETRY] {tag} attempt {attempt+1}/{max_retries}, "
                      f"waiting {wait}s... ({type(e).__name__}: {e})")
                await asyncio.sleep(wait)
                continue
            print(f"[ERROR] {tag}: {e}")
            traceback.print_exc()
            return {
                "task": sample_id,
                "model": model_label,
                "score": -1,
                "status": "ERROR",
                "steps": -1,
                "response_length": -1,
                "latency_seconds": -1,
                "duration_seconds": -1,
            }


def print_single_model_summary(model_label: str, results: list[dict], max_parallel: int):
    """打印单模型结果摘要。ERROR（基础设施故障）不计入有效评测统计。"""
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    evaluated = passed + failed
    valid_steps = [r["steps"] for r in results if r["status"] != "ERROR" and r.get("steps", -1) >= 0]
    avg_steps = sum(valid_steps) / len(valid_steps) if valid_steps else -1
    valid_len = [r["response_length"] for r in results if r["status"] != "ERROR" and r.get("response_length", -1) >= 0]
    avg_len = int(sum(valid_len) / len(valid_len)) if valid_len else -1
    valid_time = [r["duration_seconds"] for r in results if r["status"] != "ERROR" and r.get("duration_seconds", -1) >= 0]
    avg_time = sum(valid_time) / len(valid_time) if valid_time else -1
    valid_scores = [r["score"] for r in results if r["status"] != "ERROR" and r["score"] >= 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else -1
    header = f"模型: {model_label}" if model_label else "运行结果"
    print(f"\n  {header}")
    avg_steps_str = f"{avg_steps:.1f}" if avg_steps >= 0 else "N/A"
    avg_len_str = str(avg_len) if avg_len >= 0 else "N/A"
    avg_time_str = f"{avg_time:.1f}s" if avg_time >= 0 else "N/A"
    avg_score_str = f"{avg_score:.1f}" if avg_score >= 0 else "N/A"
    error_note = f" | 错误(未计入): {errors}" if errors else ""
    print(f"  有效评测: {evaluated} | 通过: {passed} | 失败: {failed}{error_note} "
          f"| 平均分: {avg_score_str} | 平均步数: {avg_steps_str} | 平均输出(计费): {avg_len_str} | 平均时延: {avg_time_str}")
    print(f"  {'-'*56}")
    for r in results:
        icon = {"PASS": "✓", "FAIL": "✗", "ERROR": "!"}[r["status"]]
        steps_str = str(r["steps"]) if r.get("steps", -1) >= 0 else "N/A"
        len_str = str(r["response_length"]) if r.get("response_length", -1) >= 0 else "N/A"
        time_str = f"{r['duration_seconds']:.1f}s" if r.get("duration_seconds", -1) >= 0 else "N/A"
        print(f"    [{icon}] {r['task']:<40s} score={r['score']:<6} steps={steps_str} 输出={len_str} time={time_str}")
    return passed, failed, errors


def compute_category_stats(results: list[dict]) -> dict[str, dict]:
    """按 PACKAGE_TASKS 类别分组统计平均分和通过率。"""
    from collections import defaultdict
    cat_results: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        cat = TASK_TO_CATEGORY.get(r["task"], "unknown")
        cat_results[cat].append(r)

    stats: dict[str, dict] = {}
    for cat in sorted(cat_results):
        items = cat_results[cat]
        evaluated = [r for r in items if r["status"] != "ERROR"]
        passed = sum(1 for r in evaluated if r["status"] == "PASS")
        failed = sum(1 for r in evaluated if r["status"] == "FAIL")
        errors = len(items) - len(evaluated)
        valid_scores = [r["score"] for r in evaluated if r["score"] >= 0]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else -1
        pass_rate = passed / len(evaluated) * 100 if evaluated else 0
        stats[cat] = {
            "total": len(items),
            "evaluated": len(evaluated),
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "avg_score": round(avg_score, 1) if avg_score >= 0 else -1,
            "pass_rate": round(pass_rate, 1),
        }
    return stats


def print_category_stats(results: list[dict]):
    """打印按类别的平均分和通过率。"""
    stats = compute_category_stats(results)
    if not stats:
        return stats
    print(f"\n{'='*70}")
    print("按类别统计")
    print(f"{'='*70}")
    print(f"  {'类别':<20s} {'评测':>4s} {'通过':>4s} {'失败':>4s} {'通过率':>7s} {'平均分':>7s}")
    print(f"  {'-'*56}")
    for cat, s in stats.items():
        avg_str = f"{s['avg_score']:.1f}" if s["avg_score"] >= 0 else "N/A"
        err_mark = f" (+{s['errors']}err)" if s["errors"] else ""
        print(f"  {cat:<20s} {s['evaluated']:>4d} {s['passed']:>4d} {s['failed']:>4d} "
              f"{s['pass_rate']:>6.1f}% {avg_str:>7s}{err_mark}")
    print(f"  {'-'*56}")
    return stats


def print_trial_summary(base_key: str, trial_summaries: list[dict],
                        result_root: str, date_str: str | None = None):
    """打印同一模型多次 trial 运行后的平均分汇总，并保存到 JSON。"""
    n_trials = len(trial_summaries)
    task_trials: dict[str, list[dict]] = {}
    for summary in trial_summaries:
        for r in summary.get("results", []):
            task_trials.setdefault(r["task"], []).append(r)

    print(f"\n{'='*80}")
    print(f"模型 {base_key} — {n_trials} 次 Trial 平均分汇总")
    print(f"{'='*80}")

    per_trial_avgs = []
    for t_idx, summary in enumerate(trial_summaries, 1):
        results = summary.get("results", [])
        valid = [r["score"] for r in results if r.get("status") != "ERROR" and r["score"] >= 0]
        avg = sum(valid) / len(valid) if valid else -1
        per_trial_avgs.append(avg)
        avg_str = f"{avg:.1f}" if avg >= 0 else "N/A"
        print(f"  Trial {t_idx}: 平均分={avg_str} (有效任务={len(valid)})")

    all_task_avgs = []
    per_task_data = {}
    for task in sorted(task_trials.keys()):
        trials = task_trials[task]
        scores = [t["score"] for t in trials]
        valid = [s for s in scores if s >= 0]
        avg = sum(valid) / len(valid) if valid else -1
        per_task_data[task] = {"scores": scores, "avg": round(avg, 2) if avg >= 0 else -1}
        if avg >= 0:
            all_task_avgs.append(avg)

    overall = sum(all_task_avgs) / len(all_task_avgs) if all_task_avgs else -1
    print(f"  总平均分: {overall:.1f}" if overall >= 0 else "  总平均分: N/A")

    avg_data = {
        "model": base_key,
        "n_trials": n_trials,
        "per_trial_avg": [round(a, 2) if a >= 0 else -1 for a in per_trial_avgs],
        "overall_avg": round(overall, 2) if overall >= 0 else -1,
        "per_task": per_task_data,
    }
    if not date_str:
        date_str = datetime.now().strftime("%Y%m%d")
    trial_dir = os.path.join(result_root, "_trial_avg")
    os.makedirs(trial_dir, exist_ok=True)
    avg_path = os.path.join(trial_dir, f"{base_key}.json")
    with open(avg_path, "w", encoding="utf-8") as f:
        json.dump(avg_data, f, ensure_ascii=False, indent=2)
    print(f"  Trial 平均分已保存到: {avg_path}")


def print_multi_model_matrix(model_keys: list[str], task_ids: list[str],
                             all_results: list[dict]):
    """打印 模型×任务 对比矩阵表格。ERROR 任务在平均计算中排除。"""
    score_map = {}
    steps_map = {}
    length_map = {}
    duration_map = {}
    status_map = {}
    for r in all_results:
        key = (r.get("model", ""), r["task"])
        score_map[key] = r["score"]
        steps_map[key] = r.get("steps", -1)
        length_map[key] = r.get("response_length", -1)
        duration_map[key] = r.get("duration_seconds", -1)
        status_map[key] = r["status"]

    model_col_width = max(len(m) for m in model_keys)
    model_col_width = max(model_col_width, 6)

    print(f"\n{'='*60}")
    print("模型 × 任务 对比矩阵 (score / steps / 输出计费长度 / 时延s)")
    print(f"{'='*60}")

    header = f"{'任务':<40s}"
    for m in model_keys:
        col_w = max(model_col_width, 14)
        header += f" | {m:>{col_w}s}"
    print(header)
    print("-" * len(header))

    for task in task_ids:
        row = f"{task:<40s}"
        for m in model_keys:
            score = score_map.get((m, task), -1)
            steps = steps_map.get((m, task), -1)
            length = length_map.get((m, task), -1)
            duration = duration_map.get((m, task), -1)
            score_str = str(int(score)) if score >= 0 else "ERR"
            steps_str = str(steps) if steps >= 0 else "-"
            len_str = str(length) if length >= 0 else "-"
            time_str = f"{duration:.1f}" if duration >= 0 else "-"
            cell = f"{score_str}/{steps_str}/{len_str}/{time_str}"
            col_w = max(model_col_width, 14)
            row += f" | {cell:>{col_w}s}"
        print(row)

    print("-" * len(header))
    avg_row = f"{'平均(排除ERR)':<40s}"
    for m in model_keys:
        valid_tasks = [t for t in task_ids if status_map.get((m, t)) != "ERROR"]
        valid_scores = [score_map.get((m, t), -1) for t in valid_tasks if score_map.get((m, t), -1) >= 0]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else -1

        valid_steps = [steps_map.get((m, t), -1) for t in valid_tasks if steps_map.get((m, t), -1) >= 0]
        avg_steps = sum(valid_steps) / len(valid_steps) if valid_steps else -1

        valid_len = [length_map.get((m, t), -1) for t in valid_tasks if length_map.get((m, t), -1) >= 0]
        avg_len = int(sum(valid_len) / len(valid_len)) if valid_len else -1

        valid_dur = [duration_map.get((m, t), -1) for t in valid_tasks if duration_map.get((m, t), -1) >= 0]
        avg_dur = sum(valid_dur) / len(valid_dur) if valid_dur else -1

        n_err = len(task_ids) - len(valid_tasks)
        score_str = f"{avg_score:.1f}" if avg_score >= 0 else "N/A"
        steps_str = f"{avg_steps:.1f}" if avg_steps >= 0 else "N/A"
        len_str = str(avg_len) if avg_len >= 0 else "N/A"
        dur_str = f"{avg_dur:.1f}" if avg_dur >= 0 else "N/A"
        cell = f"{score_str}/{steps_str}/{len_str}/{dur_str}"
        if n_err:
            cell += f"({n_err}err)"
        col_w = max(model_col_width, 14)
        avg_row += f" | {cell:>{col_w}s}"
    print(avg_row)


async def main():
    parser = argparse.ArgumentParser(description="批量运行 benchmark 任务")
    parser.add_argument("tasks", nargs="*", help="要运行的任务 ID（不指定则跑全部）。可与 --models 混用时，建议用 --tasks 指定任务避免被误解析为模型")
    parser.add_argument("--tasks", "-t", nargs="+", dest="tasks_opt", metavar="TASK",
                        help="要运行的任务 ID（如 --tasks 097-bootstrap-modify-identity-xiaolu）。与位置参数 tasks 二选一")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                        help="运行指定编号范围的任务（如 --range 1 8）")
    parser.add_argument("-p", "--parallel", type=int, default=8, metavar="N",
                        help="最大并发数（默认 4，设为 1 即串行）")
    parser.add_argument("--serial", action="store_true",
                        help="串行执行（等同于 --parallel 1）")
    parser.add_argument("-m", "--models", nargs="+", metavar="MODEL",
                        help="要评测的模型列表（如 --models qwen3.5-plus qwen-max）。"
                             "支持 MODELS 预设名 或 provider_id:model_name 自定义格式")
    parser.add_argument("--models-file", metavar="FILE",
                        help="从 JSON 文件加载模型列表，格式为 "
                             '[{"key":"名称","provider_id":"...","model":"..."}]')
    parser.add_argument("--package", nargs="+", metavar="PKG",
                        help="按分类 package 跑任务（如 --package cron dingtalk）。"
                             f"可选: {', '.join(PACKAGE_CATEGORIES)}")
    parser.add_argument("--list-packages", action="store_true",
                        help="列出所有可用 package 及其任务数并退出")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="随机打乱任务顺序（默认开启，避免固定顺序带来的偏差）")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                        help="保持任务原始顺序，不打乱")
    parser.add_argument("--retries", type=int, default=3, metavar="N",
                        help="基础设施瞬态错误（超时/500）的最大重试次数（默认 3）")
    parser.add_argument("--retry-from", metavar="RESULT_DIR",
                        help="从指定结果目录重跑 ERROR 状态的任务（读取 _batch_summary.json）")
    parser.add_argument("--retry-errors", type=int, default=0, metavar="N",
                        help="跑完后自动重试 ERROR 任务的最大轮次（默认 0 不重试）")
    parser.add_argument("--trial", type=int, default=None, metavar="N",
                        help="每个模型重复推理 N 次（整个任务集重跑），分别保留结果并计算平均分。"
                             "也可在 --models-file JSON 中为每个模型单独设置 trial 字段")
    parser.add_argument("--list-models", action="store_true",
                        help="列出所有预设模型并退出")
    args = parser.parse_args()

    if args.list_packages:
        print("可用 package 列表:")
        for pkg in PACKAGE_CATEGORIES:
            tasks = PACKAGE_TASKS[pkg]
            print(f"  {pkg:<20s}  {len(tasks)} 个任务")
            for t in tasks:
                print(f"    - {t}")
        return

    if args.list_models:
        print("预设模型列表:")
        for key, cfg in MODELS.items():
            print(f"  {key:<20s}  provider={cfg['provider_id']:<16s} model={cfg['model']}")
        return

    tee = _BatchStdoutTee()
    log_path = tee.start()
    print(f"[batch_run] 日志文件: {log_path}")
    try:
        await _run_batch_main(args)
    finally:
        tee.stop()


async def _run_batch_main(args):
    # --- 解析模型列表 ---
    model_keys: list[str] = []
    model_configs: dict[str, dict] = {}
    model_trial_counts: dict[str, int] = {}

    if args.models_file:
        with open(args.models_file, "r", encoding="utf-8") as f:
            model_list = json.load(f)
        for item in model_list:
            key = item["key"]
            trial_count = item.get("inference_trials", item.get("trial", 1))
            model_keys.append(key)
            model_configs[key] = build_provider_config_from_json(item)
            model_trial_counts[key] = trial_count
    elif args.models:
        for m in args.models:
            model_keys.append(m)
            model_configs[m] = build_provider_config(m)
            model_trial_counts[m] = 1

    # --trial CLI 参数全局覆盖
    if args.trial is not None:
        for k in model_trial_counts:
            model_trial_counts[k] = args.trial

    # 显式指定了模型（哪怕只有 1 个）就视为多模型模式：按模型名建子目录、打印矩阵
    multi_model = bool(args.models or args.models_file)

    if not model_keys:
        model_keys = [DEFAULT_MODEL]
        model_configs = {DEFAULT_MODEL: build_provider_config(DEFAULT_MODEL)}
        model_trial_counts = {DEFAULT_MODEL: args.trial or 1}

    # 展开 trial：trial > 1 时为每个模型生成 key_t1, key_t2, ... 的子 key
    base_model_keys = list(model_keys)
    expanded_model_keys: list[str] = []
    expanded_model_configs: dict[str, dict] = {}
    trial_groups: dict[str, list[str]] = {}
    for key in base_model_keys:
        tc = model_trial_counts.get(key, 1)
        if tc > 1:
            trial_keys = [f"{key}_t{t}" for t in range(1, tc + 1)]
            trial_groups[key] = trial_keys
            for tk in trial_keys:
                expanded_model_keys.append(tk)
                expanded_model_configs[tk] = model_configs[key]
        else:
            expanded_model_keys.append(key)
            expanded_model_configs[key] = model_configs[key]

    has_trials = bool(trial_groups)
    model_keys = expanded_model_keys
    model_configs = expanded_model_configs

    max_parallel = 1 if args.serial else args.parallel

    # --- retry-from 模式：从已有结果目录重跑 ERROR 任务 ---
    if args.retry_from:
        result_dir = args.retry_from
        summary, error_pairs = load_error_tasks_from_dir(result_dir)
        if not error_pairs:
            print("没有 ERROR 状态的任务需要重试")
            return

        run_ts = summary["timestamp"]
        orig_model_keys = summary.get("models", [])
        orig_multi_model = "model_stats" in summary or len(orig_model_keys) > 1
        all_task_ids = sorted(set(r["task"] for r in summary["results"]))

        if not args.models and not args.models_file:
            model_keys = orig_model_keys
            model_configs = {k: build_provider_config(k) for k in model_keys}

        retry_jobs = []
        skipped = []
        for model_label, task_id in error_pairs:
            effective_key = model_label if model_label else model_keys[0]
            if effective_key in model_configs:
                retry_jobs.append((model_label, task_id, model_configs[effective_key]))
            else:
                skipped.append((model_label, task_id))

        if skipped:
            print(f"  跳过 {len(skipped)} 个任务（模型配置缺失）: "
                  f"{[(m, t) for m, t in skipped[:5]]}...")
        if not retry_jobs:
            print("没有可执行的重试任务（所有 ERROR 模型均缺少配置）")
            return

        date_prefix = os.environ.get("RESULT_DATE_PREFIX", datetime.now().strftime("%Y%m%d"))
        result_root = os.path.join("result", date_prefix)

        print(f"重试模式: {result_dir}")
        print(f"原始 ERROR: {len(error_pairs)}, 可重试: {len(retry_jobs)}")
        print(f"最大并发: {max_parallel}")

        max_retries = max(args.retry_errors, 1)
        all_results = summary["results"]

        for retry_round in range(1, max_retries + 1):
            current_errors = [
                (ml, tid, cfg) for ml, tid, cfg in retry_jobs
                if any(r.get("model", "") == ml and r["task"] == tid
                       and r["status"] == "ERROR" for r in all_results)
            ]
            if not current_errors:
                break

            if max_retries > 1:
                print(f"\n{'='*60}")
                print(f"[RETRY {retry_round}/{max_retries}] {len(current_errors)} 个 ERROR 任务")
                print(f"{'='*60}")

            if max_parallel <= 1:
                retry_results = []
                for ml, tid, cfg in current_errors:
                    r = await run_single_task(tid, run_ts,
                                              provider_config=cfg, model_label=ml,
                                              result_root=result_root,
                                              max_retries=args.retries)
                    retry_results.append(r)
            else:
                sem = asyncio.Semaphore(max_parallel)
                coros = [
                    run_single_task(tid, run_ts,
                                    provider_config=cfg, model_label=ml,
                                    semaphore=sem,
                                    result_root=result_root,
                                    max_retries=args.retries)
                    for ml, tid, cfg in current_errors
                ]
                retry_results = list(await asyncio.gather(*coros))

            all_results = merge_retry_results(all_results, retry_results)

            new_errors = sum(1 for r in all_results if r["status"] == "ERROR")
            fixed = len(error_pairs) - new_errors
            print(f"\n[RETRY {retry_round}] 修复: {fixed}/{len(error_pairs)}, "
                  f"剩余 ERROR: {new_errors}")

        for mk in (orig_model_keys if orig_multi_model else [""]):
            model_results = [r for r in all_results if r.get("model", "") == mk]
            if model_results:
                print_single_model_summary(mk, model_results, max_parallel)

        print_category_stats(all_results)

        batch_dir = os.path.join(result_root, run_ts)
        save_batch_summary(batch_dir, all_results, orig_model_keys, all_task_ids,
                           max_parallel, run_ts, orig_multi_model)

        # --- 更新 Trial 平均分汇总 ---
        inferred_trials = infer_trial_groups(orig_model_keys)
        if inferred_trials:
            for base_key, trial_key_list in inferred_trials.items():
                trial_summaries = []
                for tk in trial_key_list:
                    trial_results = [r for r in all_results if r.get("model", "") == tk]
                    if trial_results:
                        trial_summaries.append({"results": trial_results})
                if len(trial_summaries) > 1:
                    print_trial_summary(base_key, trial_summaries,
                                        result_root=result_root,
                                        date_str=date_prefix)
        return

    # --- 解析任务列表 ---
    if args.package:
        task_ids = get_tasks_from_packages(args.package)
        if not task_ids:
            print(f"[ERROR] 未在 package {args.package} 中找到任何任务")
            return
    elif args.range:
        start, end = args.range
        task_ids = [t for t in ALL_TASKS if t.split("-")[0].isdigit() and start <= int(t.split("-")[0]) <= end]
    elif args.tasks_opt:
        task_ids = args.tasks_opt
    elif args.tasks:
        task_ids = args.tasks
    else:
        task_ids = ALL_TASKS

    if args.shuffle:
        random.shuffle(task_ids)

    total_jobs = len(model_keys) * len(task_ids)

    if has_trials:
        trial_info = ", ".join(f"{k}×{len(v)}" for k, v in trial_groups.items())
        print(f"模型(含trial展开): {', '.join(model_keys)}")
        print(f"Trial 配置: {trial_info}")
    else:
        print(f"模型: {', '.join(model_keys)}")
    print(f"任务: {len(task_ids)} 个")
    print(f"总作业数: {total_jobs} (模型 {len(model_keys)} × 任务 {len(task_ids)})")
    print(f"最大并发: {max_parallel}")
    print(f"任务列表: {', '.join(task_ids)}")

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_prefix = os.environ.get("RESULT_DATE_PREFIX", datetime.now().strftime("%Y%m%d"))
    result_root = os.path.join("result", date_prefix)

    # 构建全部 (model, task) 作业对，打散后避免同一 task 的多模型扎堆调度
    jobs = [(sample_id, model_key)
            for sample_id in task_ids
            for model_key in model_keys]
    if args.shuffle:
        random.shuffle(jobs)

    if max_parallel <= 1:
        all_results = []
        for sample_id, model_key in jobs:
            result = await run_single_task(
                sample_id, run_ts,
                provider_config=model_configs[model_key],
                model_label=model_key if multi_model else "",
                result_root=result_root,
                max_retries=args.retries,
            )
            all_results.append(result)
    else:
        semaphore = asyncio.Semaphore(max_parallel)
        coros = [
            run_single_task(
                sample_id, run_ts,
                provider_config=model_configs[model_key],
                model_label=model_key if multi_model else "",
                semaphore=semaphore,
                result_root=result_root,
                max_retries=args.retries,
            )
            for sample_id, model_key in jobs
        ]
        all_results = list(await asyncio.gather(*coros))

    # 按 (模型顺序, 任务顺序) 排序
    model_order = {m: i for i, m in enumerate(model_keys)}
    task_order = {t: i for i, t in enumerate(task_ids)}
    all_results.sort(key=lambda r: (
        model_order.get(r.get("model", ""), 999),
        task_order.get(r["task"], 999),
    ))

    # --- 打印汇总 ---
    print(f"\n{'='*60}")
    print(f"批量运行完成 — {run_ts}")
    print(f"{'='*60}")

    total_passed = total_failed = total_errors = 0
    for model_key in model_keys:
        model_results = [r for r in all_results
                         if r.get("model", "") == (model_key if multi_model else "")]
        p, f_, e = print_single_model_summary(
            model_key if multi_model else "", model_results, max_parallel)
        total_passed += p
        total_failed += f_
        total_errors += e

    if multi_model:
        print_multi_model_matrix(model_keys, task_ids, all_results)

    total_evaluated = total_passed + total_failed
    error_note = f" | 错误(未计入): {total_errors}" if total_errors else ""
    print(f"\n有效评测: {total_evaluated} | 通过: {total_passed} | 失败: {total_failed}{error_note}")
    print(f"并发: {max_parallel}")

    print_category_stats(all_results)

    # --- 保存汇总 ---
    batch_dir = os.path.join(result_root, run_ts)
    save_batch_summary(batch_dir, all_results, model_keys, task_ids,
                       max_parallel, run_ts, multi_model)

    # --- 自动重试 ERROR ---
    if args.retry_errors > 0:
        for retry_round in range(1, args.retry_errors + 1):
            error_in_results = [
                r for r in all_results if r["status"] == "ERROR"
            ]
            if not error_in_results:
                break

            print(f"\n{'='*60}")
            print(f"[AUTO-RETRY {retry_round}/{args.retry_errors}] "
                  f"重跑 {len(error_in_results)} 个 ERROR 任务")
            print(f"{'='*60}")

            retry_jobs = []
            for r in error_in_results:
                ml = r.get("model", "")
                effective_key = ml if ml else model_keys[0]
                if effective_key in model_configs:
                    retry_jobs.append((ml, r["task"], model_configs[effective_key]))
                else:
                    print(f"  [WARN] 跳过 {r['task']}: 模型 {effective_key} 无配置")

            if not retry_jobs:
                break

            if max_parallel <= 1:
                retry_results = []
                for ml, tid, cfg in retry_jobs:
                    ret = await run_single_task(tid, run_ts,
                                                provider_config=cfg,
                                                model_label=ml,
                                                result_root=result_root,
                                                max_retries=args.retries)
                    retry_results.append(ret)
            else:
                sem = asyncio.Semaphore(max_parallel)
                coros = [
                    run_single_task(tid, run_ts,
                                    provider_config=cfg, model_label=ml,
                                    semaphore=sem,
                                    result_root=result_root,
                                    max_retries=args.retries)
                    for ml, tid, cfg in retry_jobs
                ]
                retry_results = list(await asyncio.gather(*coros))

            all_results = merge_retry_results(all_results, retry_results)

            new_errors = sum(1 for r in all_results if r["status"] == "ERROR")
            fixed = len(error_in_results) - new_errors
            print(f"[AUTO-RETRY {retry_round}] 修复: {fixed}/{len(error_in_results)}, "
                  f"剩余 ERROR: {new_errors}")

            save_batch_summary(batch_dir, all_results, model_keys, task_ids,
                               max_parallel, run_ts, multi_model)

    # --- Trial 平均分汇总 ---
    if has_trials:
        date_prefix = os.environ.get("RESULT_DATE_PREFIX", datetime.now().strftime("%Y%m%d"))
        for base_key, trial_key_list in trial_groups.items():
            trial_summaries = []
            for tk in trial_key_list:
                trial_results = [r for r in all_results if r.get("model", "") == tk]
                if trial_results:
                    trial_summaries.append({"results": trial_results})
            if len(trial_summaries) > 1:
                print_trial_summary(base_key, trial_summaries,
                                    result_root=os.path.join("result", date_prefix),
                                    date_str=date_prefix)


if __name__ == "__main__":
    asyncio.run(main())
