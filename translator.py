
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


class Translator:
    """
    HY-MT1.5 offline translator.

    Automatically downloads:
        - llama.cpp CPU server from GitHub Releases
        - HY-MT1.5 GGUF model from Hugging Face

    Python dependencies: standard library only.

    Usage:
        with Translator() as t:
            print(t.translate("Hello world"))

    Environment variables:
        HYMT_CACHE       Cache directory
        HYMT_THREADS     CPU threads
        HYMT_CTX         Context size
        HF_ENDPOINT      Hugging Face mirror
    """

    HF_REPO = "tencent/HY-MT1.5-1.8B-GGUF"
    LLAMA_REPO = "ggml-org/llama.cpp"

    MODELS = {
        "Q4_K_M": "HY-MT1.5-1.8B-Q4_K_M.gguf",
        "Q6_K": "HY-MT1.5-1.8B-Q6_K.gguf",
        "Q8_0": "HY-MT1.5-1.8B-Q8_0.gguf",
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
    # Platform detection
    # --------------------------------------------------

    @staticmethod
    def _platform_asset():
        arch = platform.machine().lower()

        if os.name == "nt":
            if arch in ("amd64", "x86_64"):
                return "win-cpu-x64.zip", "zip"
            if arch in ("arm64", "aarch64"):
                return "win-cpu-arm64.zip", "zip"

        elif platform.system() == "Darwin":
            if arch in ("arm64", "aarch64"):
                return "macos-arm64.tar.gz", "tar.gz"
            if arch in ("amd64", "x86_64"):
                return "macos-x64.tar.gz", "tar.gz"

        elif platform.system() == "Linux":
            if arch in ("amd64", "x86_64"):
                return "ubuntu-x64.tar.gz", "tar.gz"
            if arch in ("arm64", "aarch64"):
                return "ubuntu-arm64.tar.gz", "tar.gz"

        raise RuntimeError(
            f"Unsupported platform: "
            f"{platform.system()}/{arch}"
        )

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

        suffix, archive_type = self._platform_asset()

        # Search recent releases instead of relying on /latest.
        api = (
            f"https://api.github.com/repos/"
            f"{self.LLAMA_REPO}/releases?per_page=100&page=1"
        )

        request = urllib.request.Request(
            api,
            headers={
                "User-Agent": "HY-MT-Python-Translator/1.0",
                "Accept": "application/vnd.github+json",
            },
        )

        with urllib.request.urlopen(
            request, timeout=30
        ) as response:
            releases = json.load(response)

        selected = None

        for release in releases:
            if release.get("draft"):
                continue

            for asset in release.get("assets", []):
                if asset["name"].endswith(suffix):
                    selected = (release, asset)
                    break

            if selected:
                break

        if not selected:
            raise RuntimeError(
                f"No matching asset found in recent releases: {suffix}"
            )

        release, asset = selected

        self.server_dir.mkdir(
            parents=True, exist_ok=True
        )

        archive = self.server_dir / asset["name"]

        print(
            f"Installing llama.cpp: "
            f"{release['tag_name']} ({asset['name']})"
        )

        self._download(
            asset["browser_download_url"],
            archive,
        )

        try:
            if archive_type == "zip":
                import zipfile

                with zipfile.ZipFile(archive) as z:
                    root = self.server_dir.resolve()

                    for member in z.infolist():
                        target = (
                            root / member.filename
                        ).resolve()

                        if (
                            target != root
                            and root not in target.parents
                        ):
                            raise RuntimeError(
                                "Unsafe archive path"
                            )

                    z.extractall(root)

            else:
                with tarfile.open(
                    archive, "r:gz"
                ) as tar:
                    root = self.server_dir.resolve()

                    for member in tar.getmembers():
                        target = (
                            root / member.name
                        ).resolve()

                        if (
                            target != root
                            and root not in target.parents
                        ):
                            raise RuntimeError(
                                "Unsafe archive path"
                            )

                        if member.issym() or member.islnk():
                            raise RuntimeError(
                                "Unexpected archive link"
                            )

                    tar.extractall(root)

        finally:
            archive.unlink(missing_ok=True)

        self.server_path = self._find_server()

        if not self.server_path:
            raise RuntimeError(
                "llama-server not found after extraction"
            )

        if os.name != "nt":
            self.server_path.chmod(
                self.server_path.stat().st_mode | 0o111
            )

        print(
            f"llama-server installed: {self.server_path}"
        )


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
        temperature=0.3,
        max_tokens=512,
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
            "top_p": 0.9,
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
