#!/usr/bin/env python3
"""
auto_eval.py

自动化评测流程：依次启动 vLLM 部署模型 → 跑 batch_run.py → kill vLLM → 下一个模型。

用法:
  # 通过 JSON 配置文件指定模型列表
  python auto_eval.py --config eval_models.json

  # 指定要跑的 package
  python auto_eval.py --config eval_models.json --package bootstrap cron file_processing himalaya search memory other QA --parallel 8

  # 指定要跑的任务
  python auto_eval.py --config eval_models.json --tasks 001-team-building-cities 002-resume

  # 指定并发数
  python auto_eval.py --config eval_models.json --parallel 8

  # 指定使用哪些 GPU（默认自动检测空闲 GPU）
  python auto_eval.py --config eval_models.json --gpus 2,3,4,5,6,7


nohup env PYTHONUNBUFFERED=1 python auto_eval_v2.py --config eval_basemodel.json --package file_processing search QA gov bootstrap memory cron other  --parallel 16 &

kill -9 1043538 2>/dev/null; pkill -9 -f "auto_eval" 2>/dev/null; pkill -9 -f "VLLM::" 2>/dev/null; pkill -9 -f "batch_run" 2>/dev/null; echo "已发送终止信号"

日志目录结构:
  logs/
    20260321/
      auto_eval_184005.log          # auto_eval 主日志
      vllm_logs/
        local-4b_184010.log         # vLLM 进程日志
  result/
    20260321/
      20260321_184100/              # batch_run 结果
        model_key/task_id/
          summary.json
          session.json


配置文件格式 (eval_models.json):
[
    {
        "key": "local-2b-sft",
        "model_path": "/path/to/Qwen3.5-2B-lr1e-5",
        "tp": 1,
        "dp": 2,
        "dtype": "bfloat16",
        "tool_call_parser": "qwen3_xml",
        "extra_args": [],
        "inference_trials": 3
    },
    {
        "key": "local-4b",
        "model_path": "/path/to/Qwen3.5-4B",
        "tp": 1,
        "dp": 4
    }
]

字段说明:
  inference_trials — 可选，同一模型重复推理 n 次（整个任务重跑），分别保留结果并计算平均分（默认 1）
                     注意：与 copaw_eval.py 中的 grading_trials（同一结果 LLM 评分多次取中位数）是不同层面的概念
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEALTH_TIMEOUT = 1200  # 最多等 10 分钟让 vLLM 启动
HEALTH_INTERVAL = 5

DEFAULT_PORT = 29000
HOST_IP = subprocess.getoutput("hostname -I").strip().split()[0]


VLLM_DIR = os.path.join(sys.exec_prefix, "bin", "vllm")

# 按天归档的日志/结果根目录
LOG_BASE = os.path.join(SCRIPT_DIR, "logs")
RESULT_BASE = os.path.join(SCRIPT_DIR, "result")


def _get_date_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def setup_logging(date_str: str, start_ts: str) -> str:
    """设置日志：同时输出到 console 和 logs/{date}/auto_eval_{time}.log。"""
    log_day_dir = os.path.join(LOG_BASE, date_str)
    os.makedirs(log_day_dir, exist_ok=True)
    log_file = os.path.join(log_day_dir, f"auto_eval_{start_ts}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    return log_file


log = logging.getLogger(__name__)


def get_host_ip() -> str:
    """返回沙箱容器内可访问宿主机的地址。

    沙箱 docker-compose 配置了 extra_hosts: host.docker.internal:host-gateway，
    所以容器内用 host.docker.internal 即可访问宿主机服务。
    """
    return HOST_IP


def find_free_gpus() -> list[int]:
    """通过 nvidia-smi 找出显存占用 < 1GiB 的空闲 GPU。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return []
    free = []
    for line in out.strip().splitlines():
        idx_str, mem_str = line.split(",")
        if int(mem_str.strip()) < 1024:
            free.append(int(idx_str.strip()))
    return sorted(free)


def wait_for_health(port: int, timeout: int = HEALTH_TIMEOUT) -> bool:
    """轮询 vLLM /v1/models 直到可用或超时。"""
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(HEALTH_INTERVAL)
    return False


def launch_vllm(
    model_cfg: dict, gpus: list[int], port: int, date_str: str | None = None
) -> subprocess.Popen:
    """启动 vLLM serve 进程，返回 Popen 对象。"""
    model_path = model_cfg["model_path"]
    tp = model_cfg.get("tp", 1)
    dp = model_cfg.get("dp", 1)
    dtype = model_cfg.get("dtype", "bfloat16")
    parser = model_cfg.get("tool_call_parser", "qwen3_xml")
    extra_args = model_cfg.get("extra_args", [])

    needed = tp * dp
    if len(gpus) < needed:
        raise RuntimeError(f"需要 {needed} 张 GPU (TP={tp} × DP={dp})，但只有 {len(gpus)} 张可用: {gpus}")
    use_gpus = gpus[:needed]

    max_model_len = model_cfg.get("max_model_len", 98304)

    cmd = [
        VLLM_DIR,
        "serve",
        model_path,
        "--dtype",
        dtype,
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        parser,
        "--tensor-parallel-size",
        str(tp),
        "--data-parallel-size",
        str(dp),
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--enable-prefix-caching",
    ] + extra_args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in use_gpus)

    log.info("  CUDA_VISIBLE_DEVICES=%s", env["CUDA_VISIBLE_DEVICES"])
    log.info("  %s", " ".join(cmd))

    if not date_str:
        date_str = _get_date_str()
    vllm_log_dir = os.path.join(LOG_BASE, date_str, "vllm_logs")
    os.makedirs(vllm_log_dir, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    log_file = os.path.join(vllm_log_dir, f"{model_cfg['key']}_{ts}.log")
    fout = open(log_file, "w")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=fout,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    log.info("  PID: %d, 日志: %s", proc.pid, log_file)
    return proc


def kill_vllm(proc: subprocess.Popen):
    """终止 vLLM 进程组。"""
    if proc.poll() is not None:
        return
    pgid = os.getpgid(proc.pid)
    log.info("  终止 vLLM 进程组 (PGID=%d) ...", pgid)
    os.killpg(pgid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log.info("  SIGTERM 超时，发送 SIGKILL ...")
        os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=10)
    log.info("  vLLM 已停止")


def run_batch(
    model_cfg: dict, port: int, batch_args: list[str], date_str: str | None = None
) -> int:
    """调用 batch_run.py 跑评测，返回 exit code。"""
    key = model_cfg["key"]
    model_id = model_cfg.get("model_id", model_cfg["model_path"])

    cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "batch_run.py"),
        "--models",
        key,
    ] + batch_args

    env = os.environ.copy()
    host_ip = get_host_ip()
    # 若已导出 AUTO_EVAL_BASE_URL（完整 OpenAI 兼容 Base，须含 /v1），则沿用；否则按 host:port 拼接
    if "AUTO_EVAL_BASE_URL" not in os.environ:
        env["AUTO_EVAL_BASE_URL"] = f"http://{host_ip}:{port}/v1"
    env["AUTO_EVAL_MODEL_ID"] = model_id
    env["AUTO_EVAL_MODEL_KEY"] = key
    if date_str:
        env["RESULT_DATE_PREFIX"] = date_str

    log.info("  命令: %s", " ".join(cmd))
    log.info("  AUTO_EVAL_BASE_URL=%s", env["AUTO_EVAL_BASE_URL"])
    log.info("  AUTO_EVAL_MODEL_ID=%s", model_id)
    result = subprocess.run(cmd, env=env)
    return result.returncode


def print_trial_summary(base_key: str, trial_summaries: list[dict], date_str: str | None = None):
    """打印同一模型多次 trial 运行后的平均分汇总，并保存到 JSON。"""
    n_trials = len(trial_summaries)
    task_trials: dict[str, list[dict]] = {}
    for summary in trial_summaries:
        for r in summary.get("results", []):
            task_trials.setdefault(r["task"], []).append(r)

    log.info("=" * 80)
    log.info("模型 %s — %d 次 Trial 平均分汇总", base_key, n_trials)
    log.info("=" * 80)

    per_trial_avgs = []
    for t_idx, summary in enumerate(trial_summaries, 1):
        results = summary.get("results", [])
        valid = [r["score"] for r in results if r.get("status") != "ERROR" and r["score"] >= 0]
        avg = sum(valid) / len(valid) if valid else -1
        per_trial_avgs.append(avg)
        avg_str = f"{avg:.1f}" if avg >= 0 else "N/A"
        log.info("  Trial %d: 平均分=%s (有效任务=%d)", t_idx, avg_str, len(valid))

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
    log.info("  总平均分: %s", f"{overall:.1f}" if overall >= 0 else "N/A")

    avg_data = {
        "model": base_key,
        "n_trials": n_trials,
        "per_trial_avg": [round(a, 2) if a >= 0 else -1 for a in per_trial_avgs],
        "overall_avg": round(overall, 2) if overall >= 0 else -1,
        "per_task": per_task_data,
    }
    if not date_str:
        date_str = _get_date_str()
    trial_dir = os.path.join(RESULT_BASE, date_str, "_trial_avg")
    os.makedirs(trial_dir, exist_ok=True)
    avg_path = os.path.join(trial_dir, f"{base_key}.json")
    with open(avg_path, "w", encoding="utf-8") as f:
        json.dump(avg_data, f, ensure_ascii=False, indent=2)
    log.info("  Trial 平均分已保存到: %s", avg_path)


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="自动化 vLLM 部署 + benchmark 评测流水线")
    parser.add_argument("--config", required=True, help="模型配置 JSON 文件路径")
    parser.add_argument(
        "--gpus", type=str, default=None, help="指定使用的 GPU 编号，逗号分隔（如 2,3,4,5）。不指定则自动检测空闲 GPU"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"vLLM 服务端口（默认 {DEFAULT_PORT}）"
    )
    parser.add_argument(
        "--package", nargs="+", metavar="PKG", help="传递给 batch_run.py 的 --package 参数"
    )
    parser.add_argument("--tasks", nargs="+", metavar="TASK", help="传递给 batch_run.py 的任务列表")
    parser.add_argument("-p", "--parallel", type=int, default=None, help="传递给 batch_run.py 的并发数")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        models = json.load(f)

    if not models:
        log.error("配置文件中没有模型")
        return

    # 按天归档：启动时确定日期，整个 run 内保持不变
    date_str = _get_date_str()
    start_ts = datetime.now().strftime("%H%M%S")
    log_file = setup_logging(date_str, start_ts)
    log.info("日志文件: %s", log_file)

    if args.gpus:
        gpus = [int(g) for g in args.gpus.split(",")]
    else:
        gpus = find_free_gpus()
        if not gpus:
            log.error("没有检测到空闲 GPU")
            return

    log.info("可用 GPU: %s", gpus)
    log.info("待评测模型: %d 个", len(models))
    log.info("=" * 60)

    batch_extra = []
    if args.package:
        batch_extra += ["--package"] + args.package
    if args.tasks:
        batch_extra = args.tasks + batch_extra
    if args.parallel is not None:
        batch_extra += ["--parallel", str(args.parallel)]

    results = {}
    trial_data: dict[str, list[dict]] = {}

    for i, model_cfg in enumerate(models, 1):
        key = model_cfg["key"]
        trial_count = model_cfg.get("inference_trials", model_cfg.get("trial", 1))
        port = args.port

        log.info("[%d/%d] 模型: %s (inference_trials=%d)", i, len(models), key, trial_count)
        log.info("  路径: %s", model_cfg["model_path"])
        log.info("-" * 60)

        log.info("[1/3] 启动 vLLM ...")
        try:
            proc = launch_vllm(model_cfg, gpus, port, date_str=date_str)
        except RuntimeError as e:
            log.error("  %s", e)
            results[key] = "SKIP (GPU不足)"
            continue

        log.info("[2/3] 等待 vLLM 就绪 ...")
        if not wait_for_health(port):
            log.error("  vLLM 启动超时，跳过该模型")
            kill_vllm(proc)
            results[key] = "SKIP (启动超时)"
            continue

        log.info("  vLLM 就绪!")
        log.info("[3/3] 运行 benchmark ...")

        result_day_dir = os.path.join(RESULT_BASE, date_str)
        os.makedirs(result_day_dir, exist_ok=True)

        for t in range(1, trial_count + 1):
            trial_key = f"{key}_t{t}" if trial_count > 1 else key

            if trial_count > 1:
                log.info("  === Trial %d/%d (key=%s) ===", t, trial_count, trial_key)

            trial_cfg = dict(model_cfg, key=trial_key)

            existing_dirs = (
                set(os.listdir(result_day_dir)) if os.path.isdir(result_day_dir) else set()
            )

            exit_code = run_batch(trial_cfg, port, batch_extra, date_str=date_str)
            results[trial_key] = "OK" if exit_code == 0 else f"FAIL (exit={exit_code})"

            if trial_count > 1:
                try:
                    new_dirs = (
                        (set(os.listdir(result_day_dir)) - existing_dirs)
                        if os.path.isdir(result_day_dir)
                        else set()
                    )
                    if new_dirs:
                        latest_dir = sorted(new_dirs)[-1]
                        summary_path = os.path.join(
                            result_day_dir, latest_dir, "_batch_summary.json"
                        )
                        if os.path.exists(summary_path):
                            with open(summary_path, "r", encoding="utf-8") as f:
                                trial_data.setdefault(key, []).append(json.load(f))
                            log.info("  Trial %d 结果已收集", t)
                except Exception as e:
                    log.warning("  读取 trial %d 结果失败: %s", t, e)

        log.info("  评测完成，关闭 vLLM ...")
        kill_vllm(proc)
        time.sleep(5)

    log.info("=" * 60)
    log.info("全部评测完成！汇总:")
    log.info("=" * 60)
    for key, status in results.items():
        log.info("  %-30s %s", key, status)

    for base_key, summaries in trial_data.items():
        print_trial_summary(base_key, summaries, date_str=date_str)


if __name__ == "__main__":
    main()
