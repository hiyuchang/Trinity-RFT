import json
import os
import time

import httpx
from e2b import Sandbox


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


def get_sandbox_info(sandbox_id, token, domain):
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

            # if dashboard_url:
            #     print(f"\n{Colors.HEADER}{Colors.BOLD}🌐 Dashboard URL:{Colors.ENDC}")
            #     print(f"{Colors.OKGREEN}{Colors.UNDERLINE}{dashboard_url}{Colors.ENDC}\n")
            # else:
            #     print(f"{Colors.WARNING}Dashboard URL not found in metadata{Colors.ENDC}")

            return data
        else:
            # print(f"{Colors.FAIL}Failed to get sandbox info: {resp.status_code}{Colors.ENDC}")
            return None

    except Exception as e:
        # print(f"{Colors.WARNING}Could not fetch dashboard URL: {e}{Colors.ENDC}")
        return None


def connect_sandbox(sandbox_id, token, domain):
    """Connect to existing sandbox"""
    if domain:
        os.environ["E2B_DOMAIN"] = domain
    os.environ["E2B_API_KEY"] = token

    sandbox = Sandbox.connect(sandbox_id)

    # print(f"    {Colors.OKGREEN}✓ Connected to sandbox{Colors.ENDC}")

    return sandbox


def create_sandbox(token, domain, template="sandbox7"):
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

    # print(f"    {Colors.OKGREEN}✓ Sandbox created{Colors.ENDC} (ID: {Colors.BOLD}{sandbox.sandbox_id}{Colors.ENDC})")

    return sandbox


def get_or_create_sandbox(sandbox_id, token, domain, template="sandbox7"):
    """Get existing sandbox or create new one"""
    if sandbox_id:
        # print(f"\n{Colors.OKCYAN}[1] Connecting to existing sandbox:{Colors.ENDC} {Colors.BOLD}{sandbox_id}{Colors.ENDC}")
        sandbox = connect_sandbox(sandbox_id, token, domain)
        get_sandbox_info(sandbox_id, token, domain)
        return sandbox, False
    else:
        # print(f"\n{Colors.OKCYAN}[1] Creating sandbox with template:{Colors.ENDC} {Colors.BOLD}{template}{Colors.ENDC}")
        sandbox = create_sandbox(token, domain, template)

        # print(f"\n{Colors.OKCYAN}[2] Waiting for sandbox to be ready...{Colors.ENDC}")
        max_attempts = 60
        for attempt in range(1, max_attempts + 1):
            try:
                is_running = sandbox.is_running()
                # status_str = f"{Colors.OKGREEN}Running{Colors.ENDC}" if is_running else f"{Colors.WARNING}Not Running{Colors.ENDC}"
                # print(f"    [{attempt}/{max_attempts}] Sandbox status: {status_str}")
                if is_running:
                    # print(f"    {Colors.OKGREEN}✓ Sandbox is now running!{Colors.ENDC}")
                    get_sandbox_info(sandbox.sandbox_id, token, domain)
                    return sandbox, True
            except Exception as e:
                pass
                # print(f"    [{attempt}/{max_attempts}] {Colors.WARNING}Failed to check status:{Colors.ENDC} {e}")

            if attempt < max_attempts:
                time.sleep(2)

        # print(f"    {Colors.WARNING}Warning: Sandbox did not reach Running state within timeout{Colors.ENDC}")
        get_sandbox_info(sandbox.sandbox_id, token, domain)
        return sandbox, True


def run_workflow(
    sandbox, task_id, oss_config, dashscope_api_key, api_server_url, model_path, logger
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

    # from pathlib import Path
    # run_path = Path(__file__).parent.parent / "utils" / "run.py"
    # with open(run_path, "r") as f:
    #     sandbox.files.write("/root/run.py", f)

    result = sandbox.commands.run(
        f"python run.py --task_id {task_id} --oss-prefix {oss_config['prefix']} --rl-url {api_server_url} --rl-model-id {model_path}",
        envs={
            "OSS_ACCESS_KEY_ID": oss_config["access_key_id"],
            "OSS_ACCESS_KEY_SECRET": oss_config["key_secret"],
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
