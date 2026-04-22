import json
import os
import time
from typing import Union

import httpx
from e2b import CommandExitException, NotFoundException, Sandbox


# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_sandbox_info(sandbox_id, token, domain, logger):
    """Get sandbox info via API to retrieve dashboard URL"""
    try:
        url = f"https://api.{domain}/sandboxes/{sandbox_id}"
        headers = {"X-API-KEY": token, "X-Generate-Dashboard-Url": "true"}

        resp = httpx.get(url, headers=headers, timeout=10)

        if resp.status_code == 200:
            data = resp.json()

            dashboard_url = None

            if "metadata" in data:
                metadata = data["metadata"]
                if isinstance(metadata, dict):
                    if "sandbox.alicloud.com/dashboard-url" in metadata:
                        dashboard_url = metadata["sandbox.alicloud.com/dashboard-url"]
                    elif "dashboard_url" in metadata:
                        dashboard_url = metadata["dashboard_url"]

            if not dashboard_url and "dashboard_url" in data:
                dashboard_url = data["dashboard_url"]

            if dashboard_url:
                logger.info(f"\n{Colors.HEADER}{Colors.BOLD}🌐 Dashboard URL:{Colors.ENDC}")
                logger.info(f"{Colors.OKGREEN}{Colors.UNDERLINE}{dashboard_url}{Colors.ENDC}\n")
            else:
                logger.warning(f"{Colors.WARNING}Dashboard URL not found in metadata{Colors.ENDC}")

            return data
        else:
            logger.error(
                f"{Colors.FAIL}Failed to get sandbox info: {resp.status_code}{Colors.ENDC}"
            )
            return None

    except Exception as e:
        logger.error(f"{Colors.WARNING}Could not fetch dashboard URL: {e}{Colors.ENDC}")
        return None


def connect_sandbox(sandbox_id, token, domain, logger):
    """Connect to existing sandbox"""
    if domain:
        os.environ["E2B_DOMAIN"] = domain
    os.environ["E2B_API_KEY"] = token

    sandbox = Sandbox.connect(sandbox_id)

    logger.info(f"    {Colors.OKGREEN}✓ Connected to sandbox{Colors.ENDC}")

    return sandbox


def create_sandbox(token, domain, template, logger) -> Sandbox:
    """Create sandbox with authorization header"""
    if domain:
        os.environ["E2B_DOMAIN"] = domain
    os.environ["E2B_API_KEY"] = token

    sandbox = Sandbox.create(
        template=template,
        timeout=3600,  # 1 hour ; TODO: make this configurable
        headers={
            "template": template,
        },
    )

    logger.info(
        f"    {Colors.OKGREEN}✓ Sandbox created{Colors.ENDC} (ID: {Colors.BOLD}{sandbox.sandbox_id}{Colors.ENDC})"
    )

    return sandbox


def get_or_create_sandbox(sandbox_id, token, domain, template, logger) -> Union[Sandbox, bool]:
    """Get existing sandbox or create new one"""
    if sandbox_id:
        logger.info(
            f"\n{Colors.OKCYAN}[1] Connecting to existing sandbox:{Colors.ENDC} {Colors.BOLD}{sandbox_id}{Colors.ENDC}"
        )
        sandbox = connect_sandbox(sandbox_id, token, domain, logger)
        get_sandbox_info(sandbox_id, token, domain, logger)
        return sandbox, False
    else:
        logger.info(
            f"\n{Colors.OKCYAN}[1] Creating sandbox with template:{Colors.ENDC} {Colors.BOLD}{template}{Colors.ENDC}"
        )
        sandbox = create_sandbox(token, domain, template, logger)

        logger.info(f"\n{Colors.OKCYAN}[2] Waiting for sandbox to be ready...{Colors.ENDC}")
        max_attempts = 60
        for attempt in range(1, max_attempts + 1):
            try:
                is_running = sandbox.is_running()
                status_str = (
                    f"{Colors.OKGREEN}Running{Colors.ENDC}"
                    if is_running
                    else f"{Colors.WARNING}Not Running{Colors.ENDC}"
                )
                logger.info(f"    [{attempt}/{max_attempts}] Sandbox status: {status_str}")
                if is_running:
                    logger.info(f"    {Colors.OKGREEN}✓ Sandbox is now running!{Colors.ENDC}")
                    get_sandbox_info(sandbox.sandbox_id, token, domain, logger)
                    return sandbox, True
            except Exception as e:
                logger.error(
                    f"    [{attempt}/{max_attempts}] {Colors.WARNING}Failed to check status:{Colors.ENDC} {e}"
                )

            if attempt < max_attempts:
                time.sleep(2)

        logger.warning(
            f"    {Colors.WARNING}Warning: Sandbox did not reach Running state within timeout{Colors.ENDC}"
        )
        get_sandbox_info(sandbox.sandbox_id, token, domain, logger)
        return sandbox, True


def run_workflow(
    sandbox: Sandbox, task_id, oss_config, dashscope_api_key, api_server_url, model_path, logger
):
    # Prepare sandbox before running the workflow
    # upload /app/working/config.json
    # upload bench_client.py, run.py, setup_provider.py to /root/
    # pip uninstall copaw -y
    # pip install qwenpaw==1.1.2
    # pip install oss2
    # patch /app/venv/lib/python3.11/site-packages/qwenpaw/agents/react_agent.py < /root/patch/model_trajectory.patch
    # patch /app/venv/lib/python3.11/site-packages/agentscope/model/_openai_model.py < /root/patch/openai_model.patch
    # patch /app/venv/lib/python3.11/site-packages/agentscope/model/_model_response.py < /root/patch/model_response.patch
    # qwenpaw app &

    from pathlib import Path

    run_path = Path(__file__).parent.parent / "utils" / "run.py"
    with open(run_path, "r") as f:
        sandbox.files.write("/root/run.py", f)

    export_data_path = Path(__file__).parent.parent / "utils" / "export_training_data.py"
    with open(export_data_path, "r") as f:
        sandbox.files.write("/root/export_training_data.py", f)

    result = sandbox.commands.run(
        f"python run.py --task_id {task_id} --oss-prefix {oss_config['prefix']} --rl-url {api_server_url} --rl-model-id {model_path}",
        envs={
            "OSS_ACCESS_KEY_ID": oss_config["access_key_id"],
            "OSS_ACCESS_KEY_SECRET": oss_config["access_key_secret"],
            "OSS_REGION": oss_config["region"],
            "OSS_ENDPOINT": oss_config["endpoint"],
            "OSS_BUCKET_NAME": oss_config["bucket_name"],
            "DASHSCOPE_API_KEY": dashscope_api_key,
        },
        timeout=3600,
    )
    # print(f"    {Colors.OKGREEN}✓ run.py executed (exit code: {result.exit_code}){Colors.ENDC}")
    # print(f"    Output: {result.stdout.strip()}")
    logger.info("result.stdout: %s", result.stdout.strip())
    logger.info("result.stderr: %s", result.stderr.strip())

    content = sandbox.files.read("/root/dataset.json")
    dataset = json.loads(content)
    return dataset


def run_eval_workflow(
    sandbox: Sandbox,
    task_id: str,
    oss_config: dict,
    dashscope_api_key: str,
    api_server_url: str,
    model_path: str,
    model_label: str,
    checkpoint_job_dir: str,
    logger,
):
    # from pathlib import Path
    # run_path = Path(__file__).parent.parent / "utils" / "run.py"
    # with open(run_path, "r") as f:
    #     sandbox.files.write("/root/run.py", f)

    # export_data_path = Path(__file__).parent.parent / "utils" / "export_training_data.py"
    # with open(export_data_path, "r") as f:
    #     sandbox.files.write("/root/export_training_data.py", f)

    # copaw_eval_path = Path(__file__).parent.parent / "utils" / "copaw_eval.py"
    # with open(copaw_eval_path, "r") as f:
    #     sandbox.files.write("/root/copaw_eval.py", f)

    t0 = time.perf_counter()
    try:
        result = sandbox.commands.run(
            # "pip install pytest py-openjudge pytest-asyncio && "
            f"python run.py --task_id {task_id} --oss-prefix {oss_config['prefix']} --rl-url {api_server_url} --rl-model-id {model_path} --evaluation",
            envs={
                "OSS_ACCESS_KEY_ID": oss_config["access_key_id"],
                "OSS_ACCESS_KEY_SECRET": oss_config["access_key_secret"],
                "OSS_REGION": oss_config["region"],
                "OSS_ENDPOINT": oss_config["endpoint"],
                "OSS_BUCKET_NAME": oss_config["bucket_name"],
                "DASHSCOPE_API_KEY": dashscope_api_key,
            },
            timeout=3600,
        )
        logger.info("result.stdout: %s", result.stdout.strip())
        logger.info("result.stderr: %s", result.stderr.strip())
    except CommandExitException as e:
        # logger.error(f"{Colors.FAIL}run.py exited with non-zero exit code: {e.exit_code}{Colors.ENDC}")
        # logger.error(f"{Colors.FAIL}Error output: {e.stderr.strip()}{Colors.ENDC}")
        logger.info("run.py exited with non-zero exit code: %s", e.exit_code)
        logger.info("Error stdout: %s", e.stdout.strip())
        logger.info("Error stderr: %s", e.stderr.strip())
    latency_seconds = time.perf_counter() - t0

    if model_label:
        task_dir = os.path.join(checkpoint_job_dir, model_label, task_id)
    else:
        task_dir = os.path.join(checkpoint_job_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    logger.info("Saving summary to %s", task_dir)

    summary = sandbox.files.read("/root/summary.json")
    with open(os.path.join(task_dir, "summary.json"), "w") as f:
        f.write(summary)
    summary_data = json.loads(summary)
    logger.info("summary.json content: %s", summary_data)

    session = sandbox.files.read("/root/session.json")
    with open(os.path.join(task_dir, "session.json"), "w") as f:
        f.write(session)
    # session_data = json.loads(session)

    try:
        screenshots = sandbox.files.read("/root/screenshots.zip", format="bytes")
        with open(os.path.join(task_dir, "screenshots.zip"), "wb") as f:
            f.write(screenshots)
    except NotFoundException as e:
        logger.warning(f"{Colors.WARNING}screenshots.zip not found: {e}{Colors.ENDC}")
        screenshots = None

    try:
        workspace_files = sandbox.files.read("/root/workspace_files.zip", format="bytes")
        with open(os.path.join(task_dir, "workspace_files.zip"), "wb") as f:
            f.write(workspace_files)
    except NotFoundException as e:
        logger.warning(f"{Colors.WARNING}workspace_files.zip not found: {e}{Colors.ENDC}")
        workspace_files = None

    score = summary_data.get("summary", {}).get("avg_score", -1) if summary_data else -1
    status = "PASS" if score == 100 else "FAIL" if score >= 0 else "ERROR"
    steps = -1
    duration_seconds = -1.0
    response_length = -1
    if summary_data and summary_data.get("tasks"):
        task_steps = [t.get("steps", -1) for t in summary_data["tasks"] if t.get("steps", -1) >= 0]
        if task_steps:
            steps = sum(task_steps)
        # 时延：优先用 worker 内记录的 duration_seconds，否则用本地 latency_seconds
        durs = [
            t.get("duration_seconds")
            for t in summary_data["tasks"]
            if t.get("duration_seconds") is not None
        ]
        if durs:
            duration_seconds = sum(durs)
        else:
            duration_seconds = latency_seconds
        # 产生 token 计费长度：trajectory 各步 thought 长度 + 最终回复长度
        total_chars = 0
        for t in summary_data["tasks"]:
            for step in t.get("trajectory") or []:
                total_chars += len((step.get("thought") or ""))
            total_chars += len((t.get("final_text") or ""))
        response_length = total_chars
    if duration_seconds < 0:
        duration_seconds = latency_seconds
    tag = f"[{model_path}] {task_id}" if model_path else task_id
    print(
        f"[{status}] {tag} — score: {score}, steps: {steps}, 输出(计费): {response_length}, time: {duration_seconds:.1f}s"
    )
    return {
        "task": task_id,
        "model": model_path,
        "score": score,
        "status": status,
        "steps": steps,
        "response_length": response_length,
        "latency_seconds": round(latency_seconds, 2),
        "duration_seconds": round(duration_seconds, 2) if duration_seconds >= 0 else -1,
    }
