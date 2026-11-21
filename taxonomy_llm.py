from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence


class TaxonomyLLM:
    """Plain-text taxonomy naming/classification client backed by llama.cpp.

    The model is deliberately NOT asked to emit JSON. Python owns all IDs,
    hierarchy relations, validation, caching, and serialization.

    Default GGUF: axteam/Qwen3.5-2B-GGUF, Q4_K_M.
    The caller may pass an existing llama-server executable (for example
    Translator.server_path) so llama.cpp is downloaded only once.
    """

    HF_REPO = "axteam/Qwen3.5-2B-GGUF"
    MODEL_FILE = "Qwen3.5-2B-Q4_K_M.gguf"
    DEFAULT_REVISION = "main"

    def __init__(
        self,
        server_path: str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        threads: int | None = None,
        context: int = 8192,
        port: int = 18081,
        timeout: int = 300,
        temperature: float = 0.15,
        max_tokens: int = 768,
        repo: str | None = None,
        model_file: str | None = None,
        revision: str | None = None,
    ):
        self.repo = repo or os.getenv("TAXONOMY_HF_REPO", self.HF_REPO)
        self.model_file = model_file or os.getenv(
            "TAXONOMY_MODEL_FILE", self.MODEL_FILE
        )
        self.revision = revision or os.getenv(
            "TAXONOMY_MODEL_REVISION", self.DEFAULT_REVISION
        )
        self.cache_dir = Path(
            cache_dir or os.getenv("TAXONOMY_CACHE", ".qwen35")
        ).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.cache_dir / self.model_file

        self.server_path = (
            Path(server_path).expanduser().resolve()
            if server_path
            else self._find_server()
        )
        self.server_dir = self.cache_dir / "llamacpp"
        self.threads = int(
            threads or os.getenv(
                "TAXONOMY_THREADS", str(max(1, os.cpu_count() or 4))
            )
        )
        self.context = int(os.getenv("TAXONOMY_CTX", str(context)))
        self.port = int(port)
        self.timeout = int(timeout)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.url = f"http://127.0.0.1:{self.port}"

        self.process: subprocess.Popen | None = None
        self.log_file = None
        self._owns_server = False

    @staticmethod
    def _download(url: str, destination: Path, timeout: int = 90) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + ".part")
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "AFSP-Taxonomy/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                total = int(response.headers.get("Content-Length", 0) or 0)
                downloaded = 0
                with temp.open("wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(
                                f"\r[Qwen3.5] Download {downloaded * 100 // total}%",
                                end="",
                                flush=True,
                            )
            if total and downloaded != total:
                raise RuntimeError(
                    f"Incomplete model download: {downloaded}/{total} bytes"
                )
            temp.replace(destination)
            print()
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    def _find_server(self) -> Path | None:
        names = ["llama-server.exe", "llama-server"] if os.name == "nt" else ["llama-server"]
        for name in names:
            found = shutil.which(name)
            if found:
                return Path(found).resolve()
        return None

    def _install_model(self) -> None:
        if self.model_path.is_file() and self.model_path.stat().st_size > 100_000_000:
            print(f"[Qwen3.5] Using cached model: {self.model_path}")
            return

        endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        url = (
            f"{endpoint}/{self.repo}/resolve/{self.revision}/"
            f"{self.model_file}?download=true"
        )
        print(f"[Qwen3.5] Downloading {self.repo}/{self.model_file}")
        self._download(url, self.model_path)

    def _port_open(self) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.4)
            try:
                sock.connect(("127.0.0.1", self.port))
                return True
            except OSError:
                return False

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def start(self) -> "TaxonomyLLM":
        if self._healthy():
            print(f"[Qwen3.5] Reusing llama-server at {self.url}")
            return self
        if self._port_open():
            raise RuntimeError(
                f"Port {self.port} is occupied, but it is not a healthy llama-server"
            )

        if self.server_path is None:
            raise RuntimeError(
                "llama-server executable not found. Start Translator first and pass "
                "server_path=translator.server_path."
            )
        if not self.server_path.is_file():
            raise FileNotFoundError(f"llama-server not found: {self.server_path}")

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
            "-np", "1",
            "--jinja",
            "--reasoning", "off",
        ]
        print("[Qwen3.5] Starting llama-server:", " ".join(command))
        self.process = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )
        self._owns_server = True

        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError(
                    f"Qwen llama-server exited. Check {log_path}"
                )
            if self._healthy():
                print("[Qwen3.5] Server ready")
                return self
            time.sleep(0.5)

        self.close()
        raise TimeoutError(f"Qwen llama-server startup timed out; check {log_path}")

    @staticmethod
    def _clean_response(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S)
        # Some server builds expose only the closing marker in message.content.
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        text = text.replace("```text", "").replace("```markdown", "").replace("```", "")
        return text.strip()

    def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self._healthy():
            raise RuntimeError("Qwen server is not running; call start() first")

        body = {
            "messages": [
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": user.strip()},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "top_p": 0.8,
            "top_k": 20,
            "repeat_penalty": 1.05,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen API error ({exc.code}): {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"Qwen request failed: {exc}") from exc

        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected llama-server response: {result!r}") from exc
        return self._clean_response(str(content))

    @staticmethod
    def _parse_numbered_lines(text: str, expected: int) -> list[str] | None:
        found: dict[int, str] = {}
        patterns = (
            re.compile(r"^\s*(\d{1,3})\s*(?:[|｜:：.)、-])\s*(.*?)\s*$"),
            re.compile(r"^\s*(\d{1,3})\s+[-–]\s+(.+?)\s*$"),
        )
        for raw_line in text.splitlines():
            line = raw_line.strip().strip("`").lstrip("-*• ").strip()
            if not line:
                continue
            match = next((p.match(line) for p in patterns if p.match(line)), None)
            if not match:
                continue
            index = int(match.group(1))
            value = match.group(2).strip().strip("`\"' *")
            value = re.sub(r"\s+", " ", value)
            value = value.rstrip(" .;；")
            if 1 <= index <= expected and value:
                found[index] = value
        if len(found) != expected or any(i not in found for i in range(1, expected + 1)):
            return None
        return [found[i] for i in range(1, expected + 1)]

    @staticmethod
    def _parse_assignments(text: str, expected: int, category_count: int) -> list[int] | None:
        found: dict[int, int] = {}
        pattern = re.compile(
            r"^\s*(\d{1,3})\s*(?:->|=>|[|:：.、])\s*(\d{1,2})\s*$"
        )
        for raw_line in text.splitlines():
            line = raw_line.strip().strip("`").lstrip("-*• ").strip()
            match = pattern.match(line)
            if not match:
                continue
            item_index = int(match.group(1))
            category_index = int(match.group(2))
            if 1 <= item_index <= expected and 1 <= category_index <= category_count:
                found[item_index] = category_index
        if len(found) != expected or any(i not in found for i in range(1, expected + 1)):
            return None
        return [found[i] for i in range(1, expected + 1)]

    def _name_chunk(self, groups: Sequence[Sequence[str]], level: str) -> list[str]:
        count = len(groups)
        blocks = []
        for index, titles in enumerate(groups, 1):
            cleaned = [str(title).strip() for title in titles if str(title).strip()]
            block = "\n".join(f"- {title}" for title in cleaned)
            blocks.append(f"[{index:03d}]\n{block or '- (empty group)'}")

        if level == "fine":
            objective = (
                "Name each small group as a concise English topic label. "
                "The titles may include books, products, habits, fears, or goals. "
                "Identify the shared subject, not merely a repeated word. "
                "Prefer 2-5 words; be specific and avoid vague labels such as General or Other."
            )
        else:
            objective = (
                "Name each group as a concise, natural English parent-topic label. "
                "The input entries are already named subtopics. Find their common broader theme. "
                "Prefer 2-6 words; do not merely concatenate the input labels."
            )

        system = (
            "You are an expert taxonomy editor. " + objective + " "
            "Output only one numbered line per input group, in the original order. "
            "Use exactly this format: 001 | English topic label. "
            "Do not output JSON, explanations, preambles, bullets, or extra lines."
        )
        user = "Name these groups:\n\n" + "\n\n".join(blocks)

        for attempt in range(2):
            response = self.complete(
                system,
                user if attempt == 0 else (
                    user + "\n\nIMPORTANT: Output exactly " + str(count) +
                    " lines numbered 001 through " + f"{count:03d}" +
                    ". Every line must be 'NNN | Title'. No other text."
                ),
                temperature=0.1 if attempt else self.temperature,
                max_tokens=max(160, count * 32),
            )
            parsed = self._parse_numbered_lines(response, count)
            if parsed:
                return parsed
            print(f"[Qwen3.5] Invalid {level} naming response; retry {attempt + 1}/2")
            print(response[:1200])

        if count > 1:
            midpoint = count // 2
            return (
                self._name_chunk(groups[:midpoint], level)
                + self._name_chunk(groups[midpoint:], level)
            )
        raise RuntimeError(
            f"Qwen could not return one valid {level} label. Response: {response[:2000]}"
        )

    def name_groups(self, groups: Sequence[Sequence[str]], level: str = "fine") -> list[str]:
        """Return exactly one English label per input title group, in order."""
        if not groups:
            return []
        return self._name_chunk(groups, level)

    def name_categories(self, topics: Sequence[tuple[str, int]], category_count: int) -> list[str]:
        """Create a small set of broad category names from (topic, count) pairs."""
        if not topics:
            raise ValueError("Cannot create categories from an empty topic list")
        category_count = min(int(category_count), len(topics))
        lines = [
            f"{index:03d} | {title} ({count} items)"
            for index, (title, count) in enumerate(topics, 1)
        ]
        system = (
            "You are designing the top level of a semantic taxonomy. "
            f"Group the supplied topics conceptually into exactly {category_count} broad, "
            "mutually distinguishable categories. The categories should cover the input "
            "without being overly broad or redundant. Give each category a concise English "
            "name of 2-5 words. Output only numbered lines using '001 | Category name'. "
            "Do not output JSON or explanations."
        )
        user = "Topics to cover:\n" + "\n".join(lines)
        response = ""
        for attempt in range(3):
            prompt = user
            if attempt:
                prompt += (
                    f"\n\nReturn exactly {category_count} lines numbered "
                    f"001 through {category_count:03d}, one category per line."
                )
            response = self.complete(
                system, prompt,
                temperature=0.1 if attempt else self.temperature,
                max_tokens=max(256, category_count * 40),
            )
            parsed = self._parse_numbered_lines(response, category_count)
            if parsed:
                return parsed
            print(f"[Qwen3.5] Invalid top-category response; retry {attempt + 1}/3")
            print(response[:1600])
        raise RuntimeError(
            f"Qwen failed to produce {category_count} top categories: {response[:2000]}"
        )

    def _assign_chunk(
        self,
        topics: Sequence[str],
        categories: Sequence[str],
    ) -> list[int]:
        lines = [f"C{index:02d} | {name}" for index, name in enumerate(categories, 1)]
        topic_lines = [f"{index:03d} | {name}" for index, name in enumerate(topics, 1)]
        system = (
            "You are assigning topics to an existing taxonomy. "
            "For each input topic, choose exactly one best-fitting category from the numbered "
            "category list. Use the category whose meaning best covers the topic. "
            "Do not create new categories. Output only lines in the form '001 -> 03', "
            "where the left number is the topic number and the right number is the category "
            "number. Output one line per topic, with no explanations or JSON."
        )
        user = (
            "Categories:\n" + "\n".join(lines)
            + "\n\nTopics:\n" + "\n".join(topic_lines)
        )
        response = ""
        for attempt in range(2):
            prompt = user
            if attempt:
                prompt += (
                    f"\n\nThere are exactly {len(topics)} topics. "
                    "Return every topic number exactly once. Format: 001 -> 03."
                )
            response = self.complete(
                system, prompt,
                temperature=0.05,
                max_tokens=max(160, len(topics) * 12),
            )
            parsed = self._parse_assignments(
                response, len(topics), len(categories)
            )
            if parsed:
                return parsed
            print("[Qwen3.5] Invalid category assignment response; retry")
            print(response[:1200])

        if len(topics) > 1:
            midpoint = len(topics) // 2
            return (
                self._assign_chunk(topics[:midpoint], categories)
                + self._assign_chunk(topics[midpoint:], categories)
            )
        raise RuntimeError(
            f"Qwen failed to assign topic '{topics[0]}': {response[:1600]}"
        )

    def assign_categories(
        self,
        topics: Sequence[str],
        categories: Sequence[str],
        batch_size: int = 12,
    ) -> list[int]:
        """Return 1-based category indices; model sees short plain-text batches."""
        if not categories:
            raise ValueError("Category list is empty")
        assignments: list[int] = []
        step = max(1, int(batch_size))
        for start in range(0, len(topics), step):
            assignments.extend(
                self._assign_chunk(topics[start:start + step], categories)
            )
        if len(assignments) != len(topics):
            raise RuntimeError("Category assignment count mismatch")
        return assignments

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
