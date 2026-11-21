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
    def _parse_single_category(text: str, category_count: int) -> int | None:
        """Parse one category code. The model is asked for A-H, not a mapping."""
        valid_letters = {
            chr(ord("A") + index): index + 1
            for index in range(min(category_count, 26))
        }
        valid_numbers = {str(index): index for index in range(1, category_count + 1)}

        for raw_line in text.splitlines():
            line = raw_line.strip().strip("`*_ ")
            if not line:
                continue
            line = re.sub(r"^(?:answer|category|类别|答案)\s*[:：-]\s*", "", line, flags=re.I)
            line = line.lstrip("-*>• ").strip()

            # Preferred protocol: one letter, optionally followed by punctuation
            # or a short explanation (e.g. "C" or "C - Health & Wellbeing").
            match = re.match(r"^([A-Z])(?:\b|[).:：|、-])", line.upper())
            if match and match.group(1) in valid_letters:
                return valid_letters[match.group(1)]

            # Be tolerant of a model that returns a numeric code instead.
            match = re.match(r"^(\d{1,2})(?:\b|[).:：|、-])", line)
            if match and match.group(1) in valid_numbers:
                return valid_numbers[match.group(1)]

            # Also accept a short natural-language answer such as "I choose B".
            # Only inspect this first meaningful line to avoid picking numbers
            # from quoted category lists or later explanations.
            letter_hits = re.findall(r"\b([A-Z])\b", line.upper())
            for letter in letter_hits:
                if letter in valid_letters:
                    return valid_letters[letter]
            number_hits = re.findall(r"\b(\d{1,2})\b", line)
            for number in number_hits:
                if number in valid_numbers:
                    return valid_numbers[number]
            break
        return None

    def _assign_one(
        self,
        topic: str,
        categories: Sequence[str],
        position: int,
        total: int,
    ) -> int:
        """Ask the small model to classify exactly one topic per request."""
        codes = [chr(ord("A") + index) for index in range(len(categories))]
        category_lines = [
            f"{code}. {name}"
            for code, name in zip(codes, categories)
        ]

        system = (
            "You are a simple topic classifier. "
            "Classify the ONE topic into the single best category. "
            "Return only its category letter, nothing else. "
            "Do not explain. Do not create categories."
        )
        user = (
            "Choose one category for this topic.\n\n"
            "Categories:\n"
            + "\n".join(category_lines)
            + "\n\nTopic:\n"
            + str(topic).strip()
            + "\n\nAnswer with exactly one letter: "
            + ", ".join(codes)
        )

        last_response = ""
        for attempt in range(3):
            prompt = user
            if attempt == 1:
                prompt = (
                    "One topic only. Pick one category letter. "
                    "Your entire answer must be a single letter.\n\n"
                    + user
                )
            elif attempt == 2:
                prompt = (
                    "This is a single-choice classification.\n"
                    + "Allowed answers: " + ", ".join(codes) + ".\n"
                    + "Return exactly one allowed letter.\n\n"
                    + "Topic: " + str(topic).strip() + "\n\n"
                    + "Categories:\n" + "\n".join(category_lines)
                )

            last_response = self.complete(
                system,
                prompt,
                temperature=0.0,
                max_tokens=8,
            )
            parsed = self._parse_single_category(
                last_response, len(categories)
            )
            if parsed is not None:
                return parsed

            print(
                f"[Qwen3.5] Invalid single-topic assignment "
                f"({position}/{total}), retry {attempt + 1}/3: "
                f"{last_response[:180]!r}"
            )

        raise RuntimeError(
            f"Qwen failed to classify topic {position}/{total}: "
            f"{topic!r}; response={last_response[:500]!r}"
        )

    def assign_categories(
        self,
        topics: Sequence[str],
        categories: Sequence[str],
        batch_size: int = 1,
    ) -> list[int]:
        """Classify topics strictly one at a time; return 1-based category IDs.

        batch_size is retained for backward compatibility and intentionally
        ignored. A 2B model performs more reliably when each request contains
        one topic and asks for one short category code.
        """
        if not categories:
            raise ValueError("Category list is empty")
        if len(categories) > 26:
            raise ValueError("Single-letter assignment supports at most 26 categories")

        assignments: list[int] = []
        total = len(topics)
        for index, topic in enumerate(topics, 1):
            assignments.append(
                self._assign_one(topic, categories, index, total)
            )
            print(
                f"[Qwen3.5] Category assignments: {index}/{total}"
            )
        return assignments


    @staticmethod
    def _clean_label(text: str) -> str:
        """Extract one concise English label from a plain-text model reply."""
        text = TaxonomyLLM._clean_response(text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""

        value = lines[0].strip().strip('`*_\"\' ')
        value = re.sub(
            r"^(?:category|label|title|name|suggested title)\s*[:：-]\s*",
            "", value, flags=re.I
        )
        value = re.sub(r"^\s*(?:[-*•]|\d+[.)、|:-])\s*", "", value)
        value = re.sub(r"\s+", " ", value).strip(" .;:—–-\"'*")
        # Remove common conversational wrappers without damaging the actual label.
        value = re.sub(
            r"^(?:the category is|the best label is|a suitable title is)\s+",
            "", value, flags=re.I
        )
        return value.strip()

    def name_group(self, titles: Sequence[str], level: str = "fine") -> str:
        """Name exactly one cluster from its member English titles.

        One cluster per request. The model returns a single plain-text label;
        Python handles identity, hierarchy and serialization.
        """
        clean_titles = [str(x).strip() for x in titles if str(x).strip()]
        if not clean_titles:
            raise ValueError("Cannot name an empty title group")

        level_hint = {
            "fine": "a specific, narrow topic",
            "parent": "a broader topic that summarizes several related subtopics",
        }.get(level, "a concise topic")
        title_lines = "\n".join(f"- {title}" for title in clean_titles)
        system = (
            "You are a concise taxonomy labeler. Read the English titles and name "
            "their shared topic. Return only one short English category name, "
            "usually 2 to 6 words. No numbering, no quotation marks, no explanation, "
            "no JSON, and no Chinese."
        )
        user = (
            f"Name {level_hint} represented by these titles:\n\n"
            f"{title_lines}\n\n"
            "Return only the English name on one line."
        )

        last = ""
        for attempt in range(3):
            prompt = user if attempt == 0 else (
                "Output exactly one concise English topic name on one line. "
                "Do not explain or enumerate.\n\n" + user
            )
            last = self.complete(
                system, prompt, temperature=0.1 if attempt == 0 else 0.0,
                max_tokens=48,
            )
            label = self._clean_label(last)
            if label and len(label) <= 90 and not label.endswith((":", ",", ";")):
                return label
            print(
                f"[Qwen3.5] Invalid {level} label, retry "
                f"{attempt + 1}/3: {last[:160]!r}"
            )

        raise RuntimeError(
            f"Qwen failed to name {level} group: {last[:400]!r}"
        )

    def name_groups(
        self, groups: Sequence[Sequence[str]], level: str = "fine"
    ) -> list[str]:
        """Backward-compatible interface; processes one group per request."""
        names = []
        for index, group in enumerate(groups, 1):
            names.append(self.name_group(group, level=level))
            print(f"[Qwen3.5] {level} labels: {index}/{len(groups)}")
        return names

    @staticmethod
    def _parse_label_lines(text: str, expected: int) -> list[str] | None:
        numbered = TaxonomyLLM._parse_numbered_lines(text, expected)
        if numbered is not None:
            return numbered

        lines = []
        for raw in text.splitlines():
            line = raw.strip().strip('`*_\"\' ')
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)、|:-])\s*", "", line)
            line = re.sub(
                r"^(?:category|label|title|name)\s*[:：-]\s*",
                "", line, flags=re.I
            )
            line = re.sub(r"\s+", " ", line).strip(" .;:—–-\"'*")
            if line:
                lines.append(line)
        if len(lines) != expected:
            return None
        if len({x.casefold() for x in lines}) != expected:
            return None
        return lines

    def name_categories(
        self, topics: Sequence[tuple[str, int] | str], category_count: int
    ) -> list[str]:
        """Create broad top-level category names from a compact topic list.

        This is a naming task, not a multi-topic assignment task. Assignment is
        done separately by assign_categories(), one topic per request.
        """
        if category_count < 1:
            raise ValueError("category_count must be positive")
        topic_lines = []
        for item in topics:
            if isinstance(item, (tuple, list)) and item:
                title = str(item[0]).strip()
                count = item[1] if len(item) > 1 else None
            else:
                title, count = str(item).strip(), None
            if not title:
                continue
            suffix = f" ({int(count)} items)" if count is not None else ""
            topic_lines.append(f"- {title}{suffix}")

        if not topic_lines:
            raise ValueError("No topics supplied for top-level category naming")

        system = (
            "You are designing a broad, mutually distinguishable taxonomy. "
            "Infer high-level domains that collectively cover the supplied topics. "
            "Return only the requested number of short English category names, "
            "one per line. Do not assign topics, explain, or output JSON."
        )
        user = (
            f"Create exactly {category_count} broad top-level category names "
            "that cover this topic list. Prefer clear, recognizable domains; "
            "avoid redundant or overlapping names.\n\n"
            + "Topics:\n" + "\n".join(topic_lines)
            + f"\n\nReturn exactly {category_count} lines, one category name per line."
        )

        last = ""
        for attempt in range(3):
            prompt = user if attempt == 0 else (
                f"Return exactly {category_count} distinct English category names, "
                "one per line. No descriptions, no assignments, no JSON.\n\n"
                + user
            )
            last = self.complete(
                system, prompt, temperature=0.1 if attempt == 0 else 0.0,
                max_tokens=max(96, category_count * 12),
            )
            names = self._parse_label_lines(last, category_count)
            if names:
                return names
            print(
                f"[Qwen3.5] Invalid top-category names, retry "
                f"{attempt + 1}/3: {last[:200]!r}"
            )

        raise RuntimeError(
            f"Qwen failed to name {category_count} top-level categories: "
            f"{last[:500]!r}"
        )

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
