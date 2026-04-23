"""
CoPaw Benchmark Client
======================

上传 / 下载 / 查看 benchmark 数据的统一接口。
数据存储在阿里云 OSS 上。

快速开始:
    from benchmark.client import BenchmarkClient

    client = BenchmarkClient()
    client.list_tasks()
    client.upload_task("my-task", "./my-task/")
    client.download_task("email-digest", "./local/")

环境变量:
    OSS_ACCESS_KEY_ID      — 阿里云 AccessKey ID
    OSS_ACCESS_KEY_SECRET  — 阿里云 AccessKey Secret
    OSS_ENDPOINT           — OSS Endpoint (默认 oss-cn-wulanchabu.aliyuncs.com)
    OSS_BUCKET_NAME        — Bucket 名称 (默认 dail-wlcb)
"""

import json
import logging
import os
import shutil
import sys
import tarfile
from pathlib import Path

import oss2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


class BenchmarkClient:
    """CoPaw Benchmark 数据操作客户端"""

    # 任务标准目录结构
    TASK_STRUCTURE = {
        "task.toml": "必须 - 任务元数据",
        "instruction.md": "必须 - 用户指令",
        "environment/": "必须 - 运行环境 (Dockerfile, data/, skills/, config/)",
        "solution/": "可选 - 参考解 (solve.sh)",
        "tests/": "可选 - 验证测试 (test.sh, test_outputs.py)",
    }

    def __init__(
        self,
        access_key_id: str = None,
        access_key_secret: str = None,
        endpoint: str = None,
        bucket_name: str = None,
        oss_prefix: str = "CoPaw-Pro/benchmark/",
    ):
        """
        初始化客户端

        Args:
            access_key_id: 阿里云 AK (或设置环境变量 OSS_ACCESS_KEY_ID)
            access_key_secret: 阿里云 SK (或设置环境变量 OSS_ACCESS_KEY_SECRET)
            endpoint: OSS Endpoint (或设置环境变量 OSS_ENDPOINT)
            bucket_name: Bucket 名称 (或设置环境变量 OSS_BUCKET_NAME)
            oss_prefix: OSS 中的根目录前缀
        """
        self.access_key_id = access_key_id or os.environ.get("OSS_ACCESS_KEY_ID")
        self.access_key_secret = access_key_secret or os.environ.get("OSS_ACCESS_KEY_SECRET")
        self.endpoint = endpoint or os.environ.get("OSS_ENDPOINT")
        self.bucket_name = bucket_name or os.environ.get("OSS_BUCKET_NAME")

        assert self.access_key_id, "No OSS_ACCESS_KEY_ID provided"
        assert self.access_key_secret, "No OSS_ACCESS_KEY_SECRET provided"
        assert self.endpoint, "No OSS_ENDPOINT provided"
        assert self.bucket_name, "No OSS_BUCKET_NAME provided"

        if oss_prefix and not oss_prefix.endswith("/"):
            oss_prefix += "/"
        self.oss_prefix = oss_prefix

        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self._bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

    # ======================== 内部工具 ========================

    def _full_key(self, key: str) -> str:
        """补全 OSS key 前缀"""
        if key.startswith(self.oss_prefix):
            return key
        return f"{self.oss_prefix}{key}"

    def _rel_key(self, full_key: str) -> str:
        """去除 OSS key 前缀"""
        if full_key.startswith(self.oss_prefix):
            return full_key[len(self.oss_prefix) :]
        return full_key

    def _safe_extract_tar(self, archive_path: Path, target_dir: Path) -> list:
        """安全解压 tar.gz，防止路径穿越。"""
        target_dir.mkdir(parents=True, exist_ok=True)
        base_dir = target_dir.resolve()

        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            member_names = [
                m.name.lstrip("./") for m in members if m.name and m.name not in (".", "./")
            ]
            top_levels = {name.split("/", 1)[0] for name in member_names if name}

            should_flatten = False
            wrapper_dir_name = None
            if len(top_levels) == 1:
                wrapper_dir_name = next(iter(top_levels))
                root_dir_exists = any(
                    m.isdir() and m.name.rstrip("/") == wrapper_dir_name for m in members
                )
                all_under_root = all(
                    name == wrapper_dir_name or name.startswith(f"{wrapper_dir_name}/")
                    for name in member_names
                )
                should_flatten = root_dir_exists and all_under_root

            for member in members:
                member_path = (target_dir / member.name).resolve()
                if os.path.commonpath([str(base_dir), str(member_path)]) != str(base_dir):
                    raise ValueError(f"压缩包包含非法路径: {member.name}")

            tar.extractall(path=target_dir)

        if should_flatten and wrapper_dir_name:
            wrapper_dir = target_dir / wrapper_dir_name
            if wrapper_dir.is_dir():
                for child in wrapper_dir.iterdir():
                    dst = target_dir / child.name
                    if dst.exists():
                        raise FileExistsError(f"解压冲突，目标已存在: {dst}")
                    shutil.move(str(child), str(dst))
                shutil.rmtree(wrapper_dir)

        extracted_files = [m.name.lstrip("./") for m in members if m.isfile()]
        if should_flatten and wrapper_dir_name:
            prefix = f"{wrapper_dir_name}/"
            extracted_files = [
                name[len(prefix) :] if name.startswith(prefix) else name for name in extracted_files
            ]
        return extracted_files

    # ======================== 上传 ========================

    def upload_task(self, task_id: str, local_dir: str) -> dict:
        """
        上传一个 benchmark 任务到 OSS

        Args:
            task_id: 任务 ID (如 "email-digest-summary")
            local_dir: 本地任务目录路径

        Returns:
            上传结果 {"task_id": ..., "files": [...], "count": ...}

        Example:
            client.upload_task("email-digest", "./tasks/email-digest/")
        """
        local_path = Path(local_dir)
        if not local_path.exists():
            raise FileNotFoundError(f"目录不存在: {local_dir}")

        # 校验必须文件
        missing = []
        for fname, desc in self.TASK_STRUCTURE.items():
            if desc.startswith("必须"):
                check_path = local_path / fname.rstrip("/")
                if not check_path.exists():
                    missing.append(fname)

        if missing:
            log.warning("缺少必须文件: %s", ", ".join(missing))
            log.warning("任务标准结构:")
            for fname, desc in self.TASK_STRUCTURE.items():
                log.warning("  %-20s %s", fname, desc)
            raise ValueError(f"任务目录不完整，缺少: {missing}")

        log.info("上传任务: %s", task_id)
        uploaded_files = []

        for root, dirs, files in os.walk(str(local_path)):
            for fname in files:
                file_path = os.path.join(root, fname)
                rel_path = os.path.relpath(file_path, str(local_path))
                oss_key = self._full_key(f"tasks/{task_id}/{rel_path}")

                self._bucket.put_object_from_file(oss_key, file_path)
                uploaded_files.append(rel_path)
                log.info("  uploaded: %s", rel_path)

        result = {
            "task_id": task_id,
            "files": uploaded_files,
            "count": len(uploaded_files),
        }
        log.info("任务 '%s' 上传完成，共 %d 个文件", task_id, len(uploaded_files))
        return result

    def upload_file(self, local_path: str, oss_key: str) -> str:
        """
        上传单个文件到 OSS

        Args:
            local_path: 本地文件路径
            oss_key: OSS 上的路径 (相对于 benchmark 根目录)

        Returns:
            完整的 oss key

        Example:
            client.upload_file("./task.toml", "tasks/my-task/task.toml")
        """
        full_key = self._full_key(oss_key)
        self._bucket.put_object_from_file(full_key, local_path)
        log.info("上传: %s -> oss://%s/%s", local_path, self.bucket_name, full_key)
        return full_key

    # ======================== 下载 ========================

    def download_task(self, task_id: str, local_dir: str) -> dict:
        """
        从 OSS 下载一个 benchmark 任务到本地

        Args:
            task_id: 任务 ID
            local_dir: 本地保存目录

        Returns:
            下载结果 {"task_id": ..., "files": [...], "count": ...}

        Example:
            client.download_task("email-digest", "./downloaded/email-digest/")
        """
        prefix = self._full_key(f"tasks/{task_id}/")
        local_path = Path(local_dir)

        log.info("下载任务: %s", task_id)
        downloaded_files = []

        for obj in oss2.ObjectIterator(self._bucket, prefix=prefix):
            if obj.key.endswith("/"):
                continue

            rel_path = obj.key[len(prefix) :]
            file_path = local_path / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)

            self._bucket.get_object_to_file(obj.key, str(file_path))
            downloaded_files.append(rel_path)
            log.info("  downloaded: %s", rel_path)

        if not downloaded_files:
            log.warning("任务 '%s' 不存在或没有文件", task_id)
            return {"task_id": task_id, "files": [], "count": 0}

        result = {
            "task_id": task_id,
            "files": downloaded_files,
            "count": len(downloaded_files),
        }
        log.info("任务 '%s' 下载完成，共 %d 个文件 -> %s", task_id, len(downloaded_files), local_dir)
        return result

    def download_compressed_task(self, task_id: str, local_dir: str) -> dict:
        """
        下载任务对应的 tar.gz 压缩包并解压到同一目录

        Args:
            task_id: 任务 ID
            local_dir: 本地保存目录

        Returns:
            下载和解压结果 {"task_id": ..., "archive": ..., "files": [...], "count": ...}

        Example:
            client.download_compressed_task("email-digest", "./downloaded/")
        """
        archive_name = f"{task_id}.tar.gz"
        candidate_keys = [
            self._full_key(f"tasks/{task_id}/{archive_name}"),
            self._full_key(f"tasks/{archive_name}"),
        ]
        local_path = Path(local_dir)
        archive_path = local_path / archive_name

        log.info("下载任务压缩包: %s", task_id)
        local_path.mkdir(parents=True, exist_ok=True)

        for full_key in candidate_keys:
            try:
                self._bucket.get_object_to_file(full_key, str(archive_path))
                break
            except oss2.exceptions.NoSuchKey:
                continue
        else:
            log.warning("任务 '%s' 的压缩包不存在: %s", task_id, archive_name)
            return {"task_id": task_id, "archive": str(archive_path), "files": [], "count": 0}

        log.info("  downloaded archive: %s", archive_path)
        extracted_files = self._safe_extract_tar(archive_path, local_path)

        result = {
            "task_id": task_id,
            "archive": str(archive_path),
            "files": extracted_files,
            "count": len(extracted_files),
        }
        log.info("任务 '%s' 压缩包下载并解压完成，共 %d 个文件 -> %s", task_id, len(extracted_files), local_dir)
        return result

    def download_file(self, oss_key: str, local_path: str) -> str:
        """
        从 OSS 下载单个文件

        Args:
            oss_key: OSS 路径 (相对于 benchmark 根目录)
            local_path: 本地保存路径

        Returns:
            本地文件路径

        Example:
            client.download_file("tasks/email-digest/instruction.md", "./instruction.md")
        """
        full_key = self._full_key(oss_key)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._bucket.get_object_to_file(full_key, local_path)
        log.info("下载: oss://%s/%s -> %s", self.bucket_name, full_key, local_path)
        return local_path

    def download_all_tasks(self, local_dir: str) -> dict:
        """
        下载所有 benchmark 任务

        Args:
            local_dir: 本地保存根目录

        Returns:
            {"tasks": [...], "total_files": ...}

        Example:
            client.download_all_tasks("./all_tasks/")
        """
        task_ids = self.list_tasks()
        total_files = 0
        for task_id in task_ids:
            result = self.download_task(task_id, os.path.join(local_dir, task_id))
            total_files += result["count"]

        log.info("全部下载完成！共 %d 个任务，%d 个文件", len(task_ids), total_files)
        return {"tasks": task_ids, "total_files": total_files}

    # ======================== 查看 ========================

    def list_tasks(self) -> list:
        """
        列出 OSS 上所有 benchmark 任务 ID

        Returns:
            任务 ID 列表

        Example:
            tasks = client.list_tasks()
            # ['email-digest-summary', 'organize-downloads', ...]
        """
        prefix = self._full_key("tasks/")
        task_ids = set()

        for obj in oss2.ObjectIterator(self._bucket, prefix=prefix):
            rel = self._rel_key(obj.key)  # tasks/{task_id}/...
            parts = rel.split("/")
            if len(parts) >= 2 and parts[0] == "tasks" and parts[1]:
                task_ids.add(parts[1])

        return sorted(task_ids)

    def list_task_files(self, task_id: str) -> list:
        """
        列出某个任务的所有文件

        Args:
            task_id: 任务 ID

        Returns:
            文件相对路径列表

        Example:
            files = client.list_task_files("email-digest")
            # ['task.toml', 'instruction.md', 'environment/Dockerfile', ...]
        """
        prefix = self._full_key(f"tasks/{task_id}/")
        files = []

        for obj in oss2.ObjectIterator(self._bucket, prefix=prefix):
            if not obj.key.endswith("/"):
                rel = obj.key[len(prefix) :]
                files.append(rel)

        return files

    def get_task_info(self, task_id: str) -> dict:
        """
        获取任务的元信息（读取 task.toml）

        Args:
            task_id: 任务 ID

        Returns:
            解析后的 task.toml 内容

        Example:
            info = client.get_task_info("email-digest")
            # {'metadata': {'difficulty': 'medium', ...}, ...}
        """
        key = self._full_key(f"tasks/{task_id}/task.toml")
        try:
            result = self._bucket.get_object(key)
            content = result.read().decode("utf-8")
            return self._parse_toml(content)
        except oss2.exceptions.NoSuchKey:
            return {}

    def get_file_content(self, oss_key: str) -> str:
        """
        读取 OSS 上的文本文件内容

        Args:
            oss_key: OSS 路径 (相对于 benchmark 根目录)

        Returns:
            文件文本内容

        Example:
            md = client.get_file_content("tasks/email-digest/instruction.md")
        """
        full_key = self._full_key(oss_key)
        result = self._bucket.get_object(full_key)
        return result.read().decode("utf-8")

    def get_sign_url(self, oss_key: str, expire: int = 7 * 24 * 3600) -> str:
        """
        生成 OSS 签名访问 URL（临时下载链接）

        Args:
            oss_key: OSS 路径
            expire: 有效期秒数 (默认 7 天)

        Returns:
            签名后的 URL

        Example:
            url = client.get_sign_url("tasks/email-digest/instruction.md")
        """
        full_key = self._full_key(oss_key)
        return self._bucket.sign_url("GET", full_key, expire)

    # ======================== 工具方法 ========================

    @staticmethod
    def _parse_toml(content: str) -> dict:
        """简单解析 TOML"""
        result = {}
        current_section = None

        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                result[current_section] = {}
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                # 去除行内注释
                if value.startswith('"'):
                    end = value.find('"', 1)
                    if end > 0:
                        value = value[1:end]
                elif value.startswith("'"):
                    end = value.find("'", 1)
                    if end > 0:
                        value = value[1:end]
                elif value.startswith("["):
                    end = value.find("]")
                    if end > 0:
                        value = [
                            v.strip().strip('"').strip("'")
                            for v in value[1:end].split(",")
                            if v.strip()
                        ]
                else:
                    if "#" in value:
                        value = value.split("#")[0].strip()
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False

                if current_section:
                    result[current_section][key] = value
                else:
                    result[key] = value

        return result

    def verify(self) -> bool:
        """
        验证 OSS 连接

        Returns:
            True 连接成功, False 失败

        Example:
            if client.verify():
                print("OSS 连接成功")
        """
        try:
            tasks = self.list_tasks()
            log.info("OSS 连接成功！")
            log.info("  Endpoint:  %s", self.endpoint)
            log.info("  Bucket:    %s", self.bucket_name)
            log.info("  Prefix:    %s", self.oss_prefix)
            log.info("  任务数量:  %d", len(tasks))
            if tasks:
                log.info("  任务列表:  %s", ", ".join(tasks))
            return True
        except Exception as e:
            log.error("OSS 连接失败: %s", e)
            return False


# ======================== CLI 命令行入口 ========================


def main():  # noqa: C901
    import argparse

    parser = argparse.ArgumentParser(
        description="CoPaw Benchmark 数据管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 验证 OSS 连接
  python client.py verify

  # 查看所有任务
  python client.py list

  # 查看某个任务的文件
  python client.py list-files email-digest

  # 上传任务
  python client.py upload email-digest ./tasks/email-digest/

  # 下载任务
  python client.py download email-digest ./downloaded/

    # 下载任务压缩包并解压
    python client.py download-compressed email-digest ./downloaded/

  # 下载所有任务
  python client.py download-all ./all_tasks/

        """,
    )
    parser.add_argument(
        "command",
        choices=[
            "verify",
            "list",
            "list-files",
            "info",
            "upload",
            "download",
            "download-compressed",
            "download-all",
            "delete",
        ],
        help="操作命令",
    )
    parser.add_argument("args", nargs="*", help="命令参数")

    args = parser.parse_args()
    client = BenchmarkClient()

    if args.command == "verify":
        client.verify()

    elif args.command == "list":
        tasks = client.list_tasks()
        log.info("共 %d 个任务:", len(tasks))
        for t in tasks:
            info = client.get_task_info(t)
            meta = info.get("metadata", {})
            diff = meta.get("difficulty", "?")
            cat = meta.get("category", "?")
            log.info("  • %-35s  [%s]  %s", t, diff, cat)

    elif args.command == "list-files":
        if not args.args:
            log.error("用法: python -m benchmark.client list-files <task_id>")
            return
        task_id = args.args[0]
        files = client.list_task_files(task_id)
        log.info("任务 '%s' 共 %d 个文件:", task_id, len(files))
        for f in files:
            log.info("  • %s", f)

    elif args.command == "info":
        if not args.args:
            log.error("用法: python -m benchmark.client info <task_id>")
            return
        task_id = args.args[0]
        info = client.get_task_info(task_id)
        log.info("%s", json.dumps(info, indent=2, ensure_ascii=False))

    elif args.command == "upload":
        if len(args.args) < 2:
            log.error("用法: python -m benchmark.client upload <task_id> <local_dir>")
            return
        client.upload_task(args.args[0], args.args[1])

    elif args.command == "download":
        if len(args.args) < 2:
            log.error("用法: python -m benchmark.client download <task_id> <local_dir>")
            return
        client.download_task(args.args[0], args.args[1])

    elif args.command == "download-compressed":
        if len(args.args) < 2:
            log.error("用法: python -m benchmark.client download-compressed <task_id> <local_dir>")
            return
        client.download_compressed_task(args.args[0], args.args[1])

    elif args.command == "download-all":
        local_dir = args.args[0] if args.args else "./benchmark_tasks/"
        client.download_all_tasks(local_dir)


if __name__ == "__main__":
    main()
