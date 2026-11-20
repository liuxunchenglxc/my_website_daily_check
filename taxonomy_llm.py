from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class TaxonomyLLMError(RuntimeError):
    """Raised when the local Qwen taxonomy model cannot be started or queried."""


class TaxonomyLLM:
    """Small local Qwen3.5-2B inference client backed by llama.cpp.

    The class deliberately uses llama-server's OpenAI-compatible endpoint and
    JSON-Schema constrained output. It can reuse the llama-server binary already
    installed by translator.py / playlist_embedding.py.

    Default model:
        axteam/Qwen3.5-2B-GGUF:Q4_K_M

    This is a text-only GGUF quantization of Qwen/Qwen3.5-2B. No vision projector
    is downloaded because the taxonomy task only uses text.
    """

    MODEL_ID = "Qwen3.5-2B"
    HF_REPO = os.getenv("QWEN_TAXONOMY_REPO", "axteam/Qwen3.5-2B-GGUF")
    HF_REVISION = os.getenv("QWEN_TAXONOMY_REVISION", "main")
    MODEL_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
    MIN_LLAMA_BUILD = 9180
    MIN_MODEL_BYTES = 1_000_000_000

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
        threads: int | None = None,
        context: int | None = None,
        port: int | None = None,
        timeout: int | None = None,
        startup_timeout: int | None = None,
        server_path: str | Path | None = None,
        hf_endpoint: str | None = None,
        model_repo: str | None = None,
        model_revision: str | None = None,
        model_file: str | None = None,
        temperature: float = 0.25,
        top_p: float = 0.9,
        top_k: int = 20,
        max_tokens: int = 2048,
    ) -> None:
        self.cache_dir = Path(
            cache_dir or os.getenv("QWEN_TAXONOMY_CACHE", ".qwen_taxonomy")
        ).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.server_dir = self.cache_dir / "llamacpp"
        self.server_dir.mkdir(parents=True, exist_ok=True)

        self.threads = int(
            threads
            or os.getenv("QWEN_TAXONOMY_THREADS", str(os.cpu_count() or 4))
        )
        self.context = int(
            context or os.getenv("QWEN_TAXONOMY_CTX", "8192")
        )
        self.port = int(
            port or os.getenv("QWEN_TAXONOMY_PORT", "18082")
        )
        self.timeout = int(
            timeout or os.getenv("QWEN_TAXONOMY_TIMEOUT", "600")
        )
        self.startup_timeout = int(startup_timeout or 240)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.max_tokens = int(max_tokens)

        self.model_repo = model_repo or self.HF_REPO
        self.model_revision = model_revision or self.HF_REVISION
        self.model_file = model_file or self.MODEL_FILE
        self.hf_endpoint = (
            hf_endpoint or os.getenv("HF_ENDPOINT", "https://huggingface.co")
        ).rstrip("/")

        configured_server = server_path or os.getenv("LLAMA_SERVER")
        self.server_path = (
            Path(configured_server).expanduser().resolve()
            if configured_server
            else None
        )
        revision_dir = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model_revision)
        self.model_path = self.cache_dir / revision_dir / self.model_file
        self.url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen[bytes] | None = None
        self.log_file = None
        self._owns_server = False
        self._server_version: str | None = None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _download(url: str, destination: Path, *, timeout: int = 120) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + ".part")
        temp.unlink(missing_ok=True)

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AFSP-Taxonomy-Generator/1.0"},
        )
        downloaded = 0
        total: int | None = None

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                header = response.headers.get("Content-Length")
                if header and header.isdigit():
                    total = int(header)
                with temp.open("wb") as file:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(
                                f"\rDownloading {destination.name}: "
                                f"{downloaded * 100 // total}%",
                                end="",
                                flush=True,
                            )

            if total is not None and downloaded != total:
                raise TaxonomyLLMError(
                    f"Incomplete download: {downloaded}/{total} bytes"
                )
            if downloaded <= 0:
                raise TaxonomyLLMError(f"Downloaded an empty file: {url}")

            temp.replace(destination)
            if total:
                print()
            print(f"Downloaded {destination.name}: {downloaded:,} bytes")
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    @staticmethod
    def _port_open(port: int) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            try:
                sock.connect(("127.0.0.1", port))
                return True
            except OSError:
                return False

    def _healthy(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.url}/health")
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def _server_matches_model(self) -> bool:
        try:
            with urllib.request.urlopen(
                f"{self.url}/v1/models", timeout=3
            ) as response:
                data = json.load(response)
            rows = data.get("data") or []
            if not rows:
                return False
            model_names = " ".join(str(row.get("id", "")) for row in rows)
            return "qwen" in model_names.lower()
        except Exception:
            return False

    def _find_server(self) -> Path | None:
        if self.server_path and self.server_path.is_file():
            return self.server_path

        env_path = os.getenv("LLAMA_SERVER")
        if env_path:
            candidate = Path(env_path).expanduser()
            if candidate.is_file():
                return candidate.resolve()

        found = shutil.which("llama-server")
        if found:
            return Path(found).resolve()

        candidate_dirs = [
            self.server_dir,
            Path(".hymt/llamacpp"),
            Path(".jina_embedding_cache/llamacpp"),
            Path(".llama/llamacpp"),
        ]
        for directory in candidate_dirs:
            if not directory.exists():
                continue
            for candidate in directory.rglob("llama-server"):
                if candidate.is_file():
                    return candidate.resolve()
        return None

    @staticmethod
    def _parse_build_number(version: str | None) -> int | None:
        if not version:
            return None
        match = re.search(r"(?:^|[^0-9])b(\d+)", version, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d{4,})", version)
        return int(match.group(1)) if match else None

    def _load_llama_release(self) -> tuple[str, str, str]:
        try:
            import llama_cpp  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TaxonomyLLMError(
                "找不到项目根目录的 llama_cpp.py，无法安装 llama-server。"
            ) from exc

        # Follow the same release selection used by translator.py and
        # playlist_embedding.py. They already use this helper in this project.
        llama_cpp.update_llama_version()
        repo = str(llama_cpp.LLAMA_REPO)
        version = str(llama_cpp.LLAMA_VERSION)
        archive = str(llama_cpp.LLAMA_FILE)
        build = self._parse_build_number(version)
        if build is not None and build < self.MIN_LLAMA_BUILD:
            raise TaxonomyLLMError(
                f"Qwen3.5 GGUF requires llama.cpp >= b{self.MIN_LLAMA_BUILD}; "
                f"current release is {version}."
            )
        return repo, version, archive

    def _install_server(self) -> None:
        existing = self._find_server()
        if existing:
            self.server_path = existing
            try:
                result = subprocess.run(
                    [str(existing), "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                version_text = (result.stdout or "") + " " + (result.stderr or "")
                match = re.search(r"b\d+", version_text, re.IGNORECASE)
                if match:
                    self._server_version = match.group(0)
                    build = self._parse_build_number(self._server_version)
                    if build is not None and build < self.MIN_LLAMA_BUILD:
                        raise TaxonomyLLMError(
                            f"llama-server {self._server_version} is too old; "
                            f"Qwen3.5 requires b{self.MIN_LLAMA_BUILD} or newer."
                        )
            except TaxonomyLLMError:
                raise
            except Exception:
                # A working binary can still be used when --version is unsupported.
                pass
            return

        if platform.system().lower() != "linux":
            raise TaxonomyLLMError(
                "自动安装 llama-server 当前仅配置了 GitHub Actions/Linux x86_64；"
                "其他平台请通过 server_path 或 LLAMA_SERVER 指定已有二进制。"
            )
        if platform.machine().lower() not in ("x86_64", "amd64"):
            raise TaxonomyLLMError(
                "自动安装 llama-server 当前仅配置了 Linux x86_64。"
            )

        repo, version, archive_name = self._load_llama_release()
        url = f"https://github.com/{repo}/releases/download/{version}/{archive_name}"
        archive_path = self.server_dir / archive_name
        print(f"Installing llama.cpp {version} for Qwen3.5...")
        self._download(url, archive_path)

        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(path=self.server_dir, filter="data")
        finally:
            archive_path.unlink(missing_ok=True)

        self.server_path = self._find_server()
        if not self.server_path:
            raise TaxonomyLLMError(
                f"llama-server was not found after extracting into {self.server_dir}"
            )
        self.server_path.chmod(self.server_path.stat().st_mode | 0o111)
        self._server_version = version

    def _install_model(self) -> None:
        # A ~1.3 GB GGUF should never be accepted as a zero-byte or partial cache.
        digest_path = self.model_path.with_suffix(self.model_path.suffix + ".sha256")
        if self.model_path.is_file() and self.model_path.stat().st_size >= self.MIN_MODEL_BYTES:
            expected_digest = digest_path.read_text(encoding="ascii").strip() if digest_path.is_file() else ""
            actual_digest = self._sha256_file(self.model_path)
            if expected_digest and actual_digest.lower() == expected_digest.lower():
                print(f"Using cached Qwen model: {self.model_path}")
                return
            print("Cached Qwen model checksum missing or mismatched; downloading again.")

        self.model_path.unlink(missing_ok=True)
        digest_path.unlink(missing_ok=True)
        endpoint = self.hf_endpoint.rstrip("/")
        url = (
            f"{endpoint}/{self.model_repo}/resolve/{self.model_revision}/"
            f"{self.model_file}?download=true"
        )
        print(
            f"Downloading Qwen3.5-2B Q4_K_M from "
            f"{self.model_repo}@{self.model_revision}..."
        )
        self._download(url, self.model_path, timeout=180)
        if self.model_path.stat().st_size < self.MIN_MODEL_BYTES:
            self.model_path.unlink(missing_ok=True)
            raise TaxonomyLLMError(
                "The downloaded Qwen GGUF is unexpectedly small; "
                "the file may be incomplete. Check HF_ENDPOINT or network access."
            )

        # Keep a local digest so a later run can detect accidental file damage.
        digest_path = self.model_path.with_suffix(self.model_path.suffix + ".sha256")
        digest_path.write_text(
            self._sha256_file(self.model_path) + "\n", encoding="ascii"
        )

    def start(self) -> "TaxonomyLLM":
        if self._healthy():
            if self._server_matches_model():
                print("Reusing running Qwen taxonomy server.")
                return self
            raise TaxonomyLLMError(
                f"Port {self.port} already serves a different model. "
                "Choose another QWEN_TAXONOMY_PORT."
            )

        if self._port_open(self.port):
            raise TaxonomyLLMError(
                f"Port {self.port} is occupied but is not a healthy Qwen server."
            )

        self._install_server()
        self._install_model()

        log_path = self.cache_dir / "llama-server.log"
        self.log_file = log_path.open("a", encoding="utf-8")
        command = [
            str(self.server_path),
            "-m", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.context),
            "-t", str(self.threads),
            "--jinja",
            "--reasoning", "off",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
        ]
        print("Starting taxonomy server:", " ".join(command))
        self.process = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        self._owns_server = True

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise TaxonomyLLMError(
                    "llama-server exited while starting Qwen3.5. "
                    f"Check {log_path}."
                )
            if self._healthy() and self._server_matches_model():
                print("Taxonomy server ready.")
                return self
            time.sleep(0.5)

        self.close()
        raise TimeoutError(
            f"Server did not become ready within {self.startup_timeout}s. "
            f"Check {log_path}."
        )

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        schema_name: str = "taxonomy_output",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        if not self._healthy():
            raise TaxonomyLLMError("Server is not running; call start() first.")

        body = {
            # llama-server already serves one loaded model; omit the model name
            # to avoid a mismatch with its generated /v1/models identifier.
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": int(max_tokens or self.max_tokens),
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        request = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TaxonomyLLMError(
                f"Qwen3.5 API error ({exc.code}): {detail[:4000]}"
            ) from exc
        except Exception as exc:
            raise TaxonomyLLMError(f"Qwen3.5 request failed: {exc}") from exc

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise TaxonomyLLMError(
                f"Unexpected llama-server response: {json.dumps(result)[:3000]}"
            ) from exc

        if isinstance(content, dict):
            parsed = content
        else:
            text = str(content or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise TaxonomyLLMError(
                    f"Returned invalid JSON: {text[:2000]}"
                ) from exc

        if not isinstance(parsed, dict):
            raise TaxonomyLLMError(
                f"Expected a JSON object, received {type(parsed).__name__}."
            )
        return parsed

    def close(self) -> None:
        if self._owns_server and self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
            self._owns_server = False

        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def __enter__(self) -> "TaxonomyLLM":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
