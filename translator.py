
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.request
import urllib.error
from pathlib import Path
import sys


class Translator:
    """
    Usage:
        with Translator() as t:
            print(t.translate("Hello world"))

    Environment variables:
        HYMT_CACHE       Cache directory
        HYMT_THREADS     CPU threads
        HYMT_CTX         Context size
        HF_ENDPOINT      Hugging Face mirror
    """

    HF_REPO = "tencent/Hy-MT2-1.8B-GGUF"
    LLAMA_REPO = "ggml-org/llama.cpp"

    MODELS = {
        "Q4_K_M": "Hy-MT2-1.8B-Q4_K_M.gguf",
        "Q6_K": "Hy-MT2-1.8B-Q6_K.gguf",
        "Q8_0": "Hy-MT2-1.8B-Q8_0.gguf",
    }

    def __init__(
        self,
        quantization="Q4_K_M",
        cache_dir=None,
        threads=None,
        context=4096,
        port=18080,
        timeout=300,
    ):
        if quantization not in self.MODELS:
            raise ValueError(
                f"Unsupported quantization: {quantization}"
            )

        self.quantization = quantization
        self.cache_dir = Path(
            cache_dir or os.getenv(
                "HYMT_CACHE", ".hymt"
            )
        ).expanduser().resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = (
            self.cache_dir / self.MODELS[quantization]
        )

        self.server_dir = self.cache_dir / "llamacpp"

        self.threads = threads or int(
            os.getenv(
                "HYMT_THREADS",
                str(max(1, (os.cpu_count() or 4) // 2))
            )
        )

        self.context = int(
            os.getenv("HYMT_CTX", str(context))
        )

        self.port = port
        self.timeout = timeout
        self.url = f"http://127.0.0.1:{port}"

        self.server_path = None
        self.process = None
        self.log_file = None

    # --------------------------------------------------
    # Download
    # --------------------------------------------------

    @staticmethod
    def _download(url, destination):
        destination = Path(destination)
        destination.parent.mkdir(
            parents=True, exist_ok=True
        )

        temp = destination.with_suffix(
            destination.suffix + ".part"
        )

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "HY-MT-Python-Translator/1.0"
            },
        )

        try:
            with urllib.request.urlopen(
                request, timeout=60
            ) as response, open(temp, "wb") as f:

                total = int(
                    response.headers.get("Content-Length", 0)
                )
                downloaded = 0

                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded += len(chunk)

                    if total:
                        print(
                            f"\rDownloading: "
                            f"{downloaded * 100 // total}%",
                            end="",
                            flush=True,
                        )

            if total and downloaded != total:
                raise RuntimeError(
                    f"Incomplete download: "
                    f"{downloaded}/{total}"
                )

            temp.replace(destination)
            print()

        except Exception:
            temp.unlink(missing_ok=True)
            raise

    # --------------------------------------------------
    # Install llama.cpp
    # --------------------------------------------------

    def _find_server(self):
        names = (
            ["llama-server.exe", "llama-server"]
            if os.name == "nt"
            else ["llama-server"]
        )

        for name in names:
            found = shutil.which(name)
            if found:
                return Path(found).resolve()

            for p in self.server_dir.rglob(name):
                if p.is_file():
                    return p.resolve()

        return None

    
    def _install_server(self):
        existing = self._find_server()
        if existing:
            self.server_path = existing
            return

        # 当前固定使用 llama.cpp b12473 Ubuntu x64 构建
        if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
            raise RuntimeError(
                "The configured llama.cpp binary only supports Linux x86_64."
            )

        url = (
            "https://github.com/ggml-org/llama.cpp/releases/download/"
            "b12473/llama-b12473-bin-ubuntu-x64.tar.gz"
        )

        self.server_dir.mkdir(parents=True, exist_ok=True)
        archive = self.server_dir / "llama-b12473-bin-ubuntu-x64.tar.gz"

        print("Downloading llama.cpp b12473...")
        self._download(url, archive)

        try:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(
                    path=self.server_dir,
                    filter="data",
                )
        finally:
            archive.unlink(missing_ok=True)

        self.server_path = self._find_server()

        if not self.server_path:
            raise RuntimeError(
                f"llama-server not found after extracting archive into {self.server_dir}"
            )

        self.server_path.chmod(
            self.server_path.stat().st_mode | 0o111
        )

        print(f"llama.cpp b12473 installed: {self.server_path}")


    # --------------------------------------------------
    # Install model
    # --------------------------------------------------

    def _install_model(self):
        if self.model_path.is_file():
            print(f"Using cached model: {self.model_path}")
            return

        filename = self.MODELS[self.quantization]

        endpoint = os.getenv(
            "HF_ENDPOINT", "https://huggingface.co"
        ).rstrip("/")

        url = (
            f"{endpoint}/{self.HF_REPO}"
            f"/resolve/main/{filename}?download=true"
        )

        print(f"Downloading model: {filename}")

        self._download(url, self.model_path)

        print(f"Model installed: {self.model_path}")

    # --------------------------------------------------
    # Server
    # --------------------------------------------------

    def _port_open(self):
        with socket.socket() as s:
            s.settimeout(0.5)

            try:
                s.connect(("127.0.0.1", self.port))
                return True
            except OSError:
                return False

    def _healthy(self):
        try:
            with urllib.request.urlopen(
                f"{self.url}/health", timeout=2
            ) as response:
                return response.status == 200
        except Exception:
            return False

    def start(self):
        if self._healthy():
            return self

        if self._port_open():
            raise RuntimeError(
                f"Port {self.port} is occupied by another service"
            )

        self._install_server()
        self._install_model()

        self.log_file = open(
            self.cache_dir / "llama-server.log",
            "a",
            encoding="utf-8",
        )

        self.process = subprocess.Popen(
            [
                str(self.server_path),
                "-m", str(self.model_path),
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "-c", str(self.context),
                "-t", str(self.threads),
            ],
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )

        deadline = time.monotonic() + 180

        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "llama-server exited unexpectedly. "
                    f"Check {self.cache_dir / 'llama-server.log'}"
                )

            if self._healthy():
                print("HY-MT1.5 server ready.")
                return self

            time.sleep(0.5)

        self.close()

        raise TimeoutError(
            "llama-server startup timed out"
        )

    # --------------------------------------------------
    # Translation
    # --------------------------------------------------

    def translate(
        self,
        text,
        target="Simplified Chinese",
        temperature=0.7,
        max_tokens=4096,
    ):
        if not text.strip():
            return ""

        if not self._healthy():
            raise RuntimeError(
                "Server is not running. Call start() first."
            )

        body = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are a professional translator. "
                        f"Translate the user's text into {target}. "
                        "Preserve every modifier, number, and "
                        "negation in the source. "
                        "Output ONLY the translated text, "
                        "with no quotation marks or commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            "temperature": temperature,
            "top_p": 0.6,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_tokens": max_tokens,
            "stream": False,
        }

        request = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(
                body, ensure_ascii=False
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json"
            },
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                result = json.load(response)

        except urllib.error.HTTPError as e:
            detail = e.read().decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(
                f"Translation API error ({e.code}): {detail}"
            ) from e

        return (
            result["choices"][0]["message"]["content"].strip()
        )

    def translate_batch(
        self,
        texts,
        target="Simplified Chinese",
        **kwargs,
    ):
        return [
            self.translate(
                text, target=target, **kwargs
            )
            for text in texts
        ]

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()

                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

            self.process = None

        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
