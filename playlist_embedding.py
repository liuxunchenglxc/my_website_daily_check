from __future__ import annotations

"""Offline semantic analysis for Innertune-style affirmation playlists.

Model
-----
Jina Embeddings v5 Text Nano - Clustering (GGUF, Q8_0)
  Repository: jinaai/jina-embeddings-v5-text-nano-clustering-GGUF
  Revision:   bc43da70719b9e941bc2094d8157e3b0ac762c17
  File:       v5-nano-clustering-Q8_0.gguf
  SHA256:     a493561322842a994eb5826e7a6167f9ea715f8e4c21757ce35abfc2b330b8d9
  Dimension:  768

The clustering model requires the exact "Document: " prefix and last-token
pooling. This class applies both automatically.

Runtime
-------
Python 3.12+
  pip install numpy scipy

The class downloads the GGUF model and (on Linux x86_64) the same llama.cpp
b12473 Ubuntu CPU archive used by the companion translator.py. You can also
provide LLAMA_SERVER or an executable already on PATH to reuse an installation.

Input JSON schema
-----------------
{
  "list": {
    "https://.../playlist-a": {"title_en": "Playlist A", ...}
  },
  "contents": {
    "https://.../playlist-a": {
      "url": "https://.../playlist-a",
      "sentences": ["I am ...", "I can ..."]
    }
  }
}

Outputs
-------
playlist_analysis.json : one medoid/core sentence and its vector per playlist
playlist_hierarchy.json: average-linkage dendrogram; every merge records the
                          left and right playlist sets and their core sentences
playlist_neighbors.json: nearest playlists by cosine similarity
manifest.json           : model and analysis metadata

License note: Jina's model is CC BY-NC 4.0. Check its terms before commercial use.
"""

import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import sys
import argparse


class PlaylistEmbedding:
    """Embed affirmations, find playlist medoids, and cluster playlists.

    Typical use::

        with PlaylistEmbedding() as analyzer:
            results = analyzer.analyze_json(
                "innertune_list.json",
                output_dir="analysis_output",
            )

    Public methods:
        embed(sentence)                 -> list[float]
        embed_batch(sentences)          -> list[list[float]]
        build_sentence_map(sentences)   -> {vector_tuple: [sentence, ...]}
        find_core_sentence(...)         -> playlist core record (and clears map)
        analyze_json(source, ...)       -> complete analysis and optional JSON files

    The per-playlist vector-to-sentence map is deliberately temporary. Once a
    playlist's medoid is selected, find_core_sentence() clears that map. Only
    the playlist core sentence and its vector are retained for global clustering.
    """

    # Pinned model revision: prevents a rerun from silently changing models.
    HF_REPO = "jinaai/jina-embeddings-v5-text-nano-clustering-GGUF"
    HF_REVISION = "bc43da70719b9e941bc2094d8157e3b0ac762c17"
    MODEL_FILES = {
        "Q8_0": {
            "filename": "v5-nano-clustering-Q8_0.gguf",
            "size": 232_883_776,
            "sha256": "a493561322842a994eb5826e7a6167f9ea715f8e4c21757ce35abfc2b330b8d9",
        },
        "Q6_K": {
            "filename": "v5-nano-clustering-Q6_K.gguf",
            "size": 181_601_344,
            "sha256": "bb40aabf8920dc33437f1cfff1ae01ca94809bd0e0303bce7d9bee006a6149ab",
        },
        "Q5_K_M": {
            "filename": "v5-nano-clustering-Q5_K_M.gguf",
            "size": 168_910_912,
            "sha256": "7ddc94f30f73805307fef03b9b9274d250ca09d75be44e4c9df79a708214cb71",
        },
        "Q4_K_M": {
            "filename": "v5-nano-clustering-Q4_K_M.gguf",
            "size": 156_966_976,
            "sha256": "1a11d4c9e81a75fd937fa97eef74edd9d69679f7942c671284ccecd9b1e5df41",
        },
    }

    LLAMA_VERSION = "b12473"
    LLAMA_REPO = "ggml-org/llama.cpp"
    LLAMA_LINUX_X64_URL = (
        "https://github.com/ggml-org/llama.cpp/releases/download/"
        "b12473/llama-b12473-bin-ubuntu-x64.tar.gz"
    )
    EMBEDDING_DIMENSION = 768
    INPUT_PREFIX = "Document: "
    POOLING = "last"

    def __init__(
        self,
        quantization: str = "Q8_0",
        cache_dir: str | Path | None = None,
        threads: int | None = None,
        context: int = 8192,
        batch_size: int = 4096,
        ubatch_size: int = 512,
        port: int = 18081,
        timeout: int = 180,
        startup_timeout: int = 180,
        auto_install_server: bool = True,
        server_path: str | Path | None = None,
        hf_endpoint: str | None = None,
    ):
        if quantization not in self.MODEL_FILES:
            raise ValueError(
                f"Unsupported quantization {quantization!r}; "
                f"choose one of {', '.join(self.MODEL_FILES)}"
            )

        self.quantization = quantization
        self.model_info = self.MODEL_FILES[quantization]
        self.cache_dir = Path(
            cache_dir or os.getenv("JINA_EMBED_CACHE", ".jina_embedding_cache")
        ).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = self.cache_dir / self.model_info["filename"]
        self.server_dir = self.cache_dir / "llamacpp"
        self.server_dir.mkdir(parents=True, exist_ok=True)

        cpu_count = os.cpu_count() or 4
        self.threads = int(threads or os.getenv("JINA_EMBED_THREADS", str(cpu_count)))
        self.context = int(context)
        self.batch_size = int(batch_size)
        self.ubatch_size = int(ubatch_size)
        self.port = int(port)
        self.timeout = int(timeout)
        self.startup_timeout = int(startup_timeout)
        self.auto_install_server = bool(auto_install_server)
        self.hf_endpoint = (
            hf_endpoint or os.getenv("HF_ENDPOINT", "https://huggingface.co")
        ).rstrip("/")
        self.url = f"http://127.0.0.1:{self.port}"

        configured_server = server_path or os.getenv("LLAMA_SERVER")
        self.server_path = Path(configured_server).expanduser().resolve() if configured_server else None
        self.process: subprocess.Popen | None = None
        self.log_file = None

        # Populated by analyze_playlists(); sentence-level maps are not retained.
        self.playlist_cores: dict[str, dict[str, Any]] = {}
        self.hierarchy: dict[str, Any] | None = None
        self.neighbors: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Download and installation
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _download(
        cls,
        url: str,
        destination: str | Path,
        expected_size: int | None = None,
        expected_sha256: str | None = None,
    ) -> Path:
        """Download atomically; validate cached files and new downloads."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.is_file() and destination.stat().st_size > 0:
            cached_size = destination.stat().st_size
            size_ok = expected_size is None or cached_size == expected_size
            hash_ok = (
                expected_sha256 is None
                or cls._sha256(destination).lower() == expected_sha256.lower()
            )
            if size_ok and hash_ok:
                print(f"Using verified cache: {destination}")
                return destination
            print(f"Cached file failed validation; downloading again: {destination}")

        temp = destination.with_name(destination.name + ".part")
        temp.unlink(missing_ok=True)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Playlist-Embedding/1.0"},
        )

        digest = hashlib.sha256()
        downloaded = 0
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temp.open("wb") as file:
                header_size = response.headers.get("Content-Length")
                total = int(header_size) if header_size and header_size.isdigit() else None
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(
                            f"\rDownloading {destination.name}: "
                            f"{downloaded * 100 // total}%",
                            end="",
                            flush=True,
                        )

            if total is not None and downloaded != total:
                raise RuntimeError(f"Incomplete download: {downloaded}/{total} bytes")
            if expected_size is not None and downloaded != expected_size:
                raise RuntimeError(
                    f"Unexpected file size: {downloaded}, expected {expected_size}"
                )
            actual_hash = digest.hexdigest()
            if expected_sha256 and actual_hash.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"SHA-256 mismatch for {destination.name}: {actual_hash}"
                )
            if downloaded <= 0:
                raise RuntimeError(f"Downloaded file is empty: {url}")

            temp.replace(destination)
            print(f"\nDownloaded: {destination} ({downloaded:,} bytes)")
            return destination
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    def _find_server(self) -> Path | None:
        if self.server_path and self.server_path.is_file():
            return self.server_path

        env_path = os.getenv("LLAMA_SERVER")
        if env_path:
            path = Path(env_path).expanduser()
            if path.is_file():
                return path.resolve()

        found = shutil.which("llama-server")
        if found:
            return Path(found).resolve()

        # Reuse the server downloaded by translator.py when available.
        candidate_dirs = [
            self.server_dir,
            Path(".hymt/llamacpp").expanduser(),
            Path(".llama/llamacpp").expanduser(),
        ]
        for directory in candidate_dirs:
            if not directory.exists():
                continue
            for candidate in directory.rglob("llama-server"):
                if candidate.is_file():
                    return candidate.resolve()
        return None

    def _install_server(self) -> None:
        existing = self._find_server()
        if existing:
            self.server_path = existing
            print(f"Using llama-server: {self.server_path}")
            return

        if not self.auto_install_server:
            raise FileNotFoundError(
                "llama-server was not found. Put it on PATH, set LLAMA_SERVER, "
                "or enable auto_install_server."
            )

        if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
            raise RuntimeError(
                "Automatic installation currently targets Linux x86_64 GitHub Actions. "
                "On another platform, set LLAMA_SERVER to an existing executable."
            )

        archive = self.server_dir / f"llama-{self.LLAMA_VERSION}-bin-ubuntu-x64.tar.gz"
        self._download(self.LLAMA_LINUX_X64_URL, archive)
        try:
            with tarfile.open(archive, "r:gz") as tar:
                try:
                    tar.extractall(path=self.server_dir, filter="data")
                except TypeError as exc:
                    raise RuntimeError(
                        "Safe tar extraction requires Python 3.12+. "
                        "Use actions/setup-python with Python 3.12 or newer."
                    ) from exc
        finally:
            archive.unlink(missing_ok=True)

        self.server_path = self._find_server()
        if not self.server_path:
            raise RuntimeError(
                f"llama-server was not found after extracting into {self.server_dir}"
            )
        self.server_path.chmod(self.server_path.stat().st_mode | 0o111)
        print(f"Installed llama-server: {self.server_path}")

    def _install_model(self) -> None:
        filename = self.model_info["filename"]
        url = (
            f"{self.hf_endpoint}/{self.HF_REPO}/resolve/"
            f"{self.HF_REVISION}/{filename}?download=true"
        )
        self._download(
            url,
            self.model_path,
            expected_size=self.model_info["size"],
            expected_sha256=self.model_info["sha256"],
        )

    # ------------------------------------------------------------------
    # llama-server lifecycle
    # ------------------------------------------------------------------

    def _port_open(self) -> bool:
        with socket.socket() as sock:
            sock.settimeout(0.5)
            try:
                sock.connect(("127.0.0.1", self.port))
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

    def start(self) -> "PlaylistEmbedding":
        if self._healthy():
            return self
        if self._port_open():
            raise RuntimeError(
                f"Port {self.port} is already occupied by a service that is not a healthy llama-server"
            )

        self._install_server()
        self._install_model()

        log_path = self.cache_dir / "llama-server.log"
        self.log_file = log_path.open("a", encoding="utf-8")
        command = [
            str(self.server_path),
            "-m", str(self.model_path),
            "--embedding",
            "--pooling", self.POOLING,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.context),
            "-b", str(self.batch_size),
            "-ub", str(self.ubatch_size),
            "-t", str(self.threads),
        ]
        print("Starting llama-server for Jina clustering embeddings...")
        self.process = subprocess.Popen(
            command,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError(
                    "llama-server exited unexpectedly. Check "
                    f"{log_path} for details."
                )
            if self._healthy():
                print(f"Embedding server ready at {self.url}")
                return self
            time.sleep(0.5)

        self.close()
        raise TimeoutError(
            f"llama-server startup timed out after {self.startup_timeout}s; "
            f"check {log_path}"
        )

    # ------------------------------------------------------------------
    # Embedding API
    # ------------------------------------------------------------------

    @classmethod
    def _format_input(cls, text: str) -> str:
        text = str(text).strip()
        if not text:
            raise ValueError("Cannot embed an empty sentence")
        if text.startswith(cls.INPUT_PREFIX):
            return text
        return cls.INPUT_PREFIX + text

    def _post_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not self._healthy():
            raise RuntimeError("Embedding server is not running. Call start() first.")

        body = {
            "input": [self._format_input(text) for text in texts],
            "model": "jina-embeddings-v5-text-nano-clustering",
            "encoding_format": "float",
        }
        request = urllib.request.Request(
            f"{self.url}/v1/embeddings",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Embedding API error ({exc.code}): {detail}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Embedding API request failed: {exc}") from exc

        rows = result.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise RuntimeError(
                "Unexpected /v1/embeddings response: expected "
                f"{len(texts)} vectors, got {len(rows) if isinstance(rows, list) else type(rows).__name__}"
            )
        rows.sort(key=lambda item: item.get("index", 0))

        vectors: list[list[float]] = []
        for row in rows:
            vector = row.get("embedding")
            if not isinstance(vector, list) or len(vector) != self.EMBEDDING_DIMENSION:
                got = len(vector) if isinstance(vector, list) else type(vector).__name__
                raise RuntimeError(
                    f"Expected {self.EMBEDDING_DIMENSION}-dimensional embedding, got {got}"
                )
            array = np.asarray(vector, dtype=np.float32)
            norm = float(np.linalg.norm(array))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise RuntimeError("Embedding model returned an invalid/zero vector")
            array /= norm
            vectors.append(array.astype(np.float32).tolist())
        return vectors

    def embed(self, sentence: str) -> list[float]:
        """Embed one English affirmation and return a normalized 768-D vector."""
        if not self._healthy():
            self.start()
        return self._post_embeddings([sentence])[0]

    def embed_batch(
        self,
        sentences: Iterable[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> list[list[float]]:
        """Embed multiple sentences, preserving input order."""
        texts = [str(sentence).strip() for sentence in sentences]
        if not texts:
            return []
        if not self._healthy():
            self.start()

        output: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, max(1, int(batch_size))):
            batch = texts[start : start + max(1, int(batch_size))]
            output.extend(self._post_embeddings(batch))
            if show_progress:
                done = min(start + len(batch), total)
                print(f"Embedded {done}/{total} sentences", flush=True)
        return output

    # ------------------------------------------------------------------
    # Playlist parsing and core sentence selection
    # ------------------------------------------------------------------

    @staticmethod
    def load_json(source: str | Path | dict[str, Any]) -> dict[str, Any]:
        """Load the Innertune JSON from a dict, local file, or HTTP(S) URL."""
        if isinstance(source, dict):
            return source
        source_text = str(source)
        if source_text.startswith(("https://", "http://")):
            request = urllib.request.Request(
                source_text,
                headers={"User-Agent": "Playlist-Embedding/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
        else:
            with Path(source).expanduser().open("r", encoding="utf-8") as file:
                data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("The root JSON value must be an object")
        return data

    @staticmethod
    def _fallback_title(url: str) -> str:
        slug = url.rstrip("/").split("/")[-1]
        return slug.replace("-", " ").replace("_", " ").title() or url

    @classmethod
    def extract_playlists(
        cls,
        data: dict[str, Any],
        include_coming_soon: bool = False,
    ) -> list[dict[str, Any]]:
        """Extract playlist records from top-level `list` and `contents` maps."""
        metadata_map = data.get("list") or {}
        content_map = data.get("contents") or data.get("content") or {}

        if not isinstance(metadata_map, dict):
            metadata_map = {}
        if not isinstance(content_map, dict):
            content_map = {}

        # Fallback for exports where sentences are directly inside `list`.
        if not content_map:
            content_map = {
                key: value
                for key, value in metadata_map.items()
                if isinstance(value, dict) and isinstance(value.get("sentences"), list)
            }

        playlists: list[dict[str, Any]] = []
        for playlist_url, content in content_map.items():
            if not isinstance(content, dict):
                continue
            metadata = metadata_map.get(playlist_url, {})
            if not isinstance(metadata, dict):
                metadata = {}

            is_coming_soon = bool(
                metadata.get("is_coming_soon", content.get("is_coming_soon", False))
            )
            if is_coming_soon and not include_coming_soon:
                continue

            sentences = content.get("sentences")
            if not isinstance(sentences, list):
                sentences = metadata.get("sentences", [])
            if not isinstance(sentences, list):
                continue

            clean_sentences = [
                str(sentence).strip()
                for sentence in sentences
                if isinstance(sentence, str) and sentence.strip()
            ]
            if not clean_sentences:
                continue

            url = str(content.get("url") or playlist_url)
            title_en = (
                metadata.get("title_en")
                or content.get("title_en")
                or cls._fallback_title(url)
            )
            title_cn = (
                metadata.get("title_cn")
                or content.get("title_cn")
                or cls._fallback_title(url)
            )
            playlists.append(
                {
                    "playlist_id": url,
                    "playlist_url": url,
                    "title_en": str(title_en),
                    "title_cn": str(title_cn),
                    "intro_en": metadata.get("intro_en") or content.get("intro_en"),
                    "intro_cn": metadata.get("intro_cn") or content.get("intro_cn"),
                    "time": content.get("time", metadata.get("time")),
                    "sentences": clean_sentences,
                }
            )
        return playlists

    def build_sentence_map(
        self,
        sentences: Iterable[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> dict[tuple[float, ...], list[str]]:
        """Build the requested vector -> sentence mapping for one playlist.

        Tuple vectors are hashable. A vector maps to a list so exact duplicate
        sentences/embeddings are preserved rather than silently overwritten.
        """
        texts = [str(sentence).strip() for sentence in sentences if str(sentence).strip()]
        vectors = self.embed_batch(texts, batch_size=batch_size, show_progress=show_progress)
        mapping: dict[tuple[float, ...], list[str]] = {}
        for sentence, vector in zip(texts, vectors):
            key = tuple(float(value) for value in vector)
            mapping.setdefault(key, []).append(sentence)
        return mapping

    @staticmethod
    def find_core_sentence(
        playlist_id: str,
        title_en: str,
        title_cn: str,
        sentence_map: dict[tuple[float, ...], list[str]],
        playlist_url: str | None = None,
        intro_en: str | None = None,
        intro_cn: str | None = None,
    ) -> dict[str, Any]:
        """Find the medoid nearest to the playlist mean and clear sentence_map.

        The core sentence is an original affirmation from the playlist, not a
        synthetic/averaged sentence. `sentence_map.clear()` releases all temporary
        sentence vectors immediately after the core has been chosen.
        """
        if not sentence_map:
            raise ValueError(f"Playlist {playlist_id!r} has no sentence embeddings")

        vectors: list[tuple[float, ...]] = []
        sentences: list[str] = []
        for vector_key, mapped_sentences in sentence_map.items():
            for sentence in mapped_sentences:
                vectors.append(vector_key)
                sentences.append(sentence)

        matrix = np.asarray(vectors, dtype=np.float32)
        centroid = matrix.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm <= 1e-12:
            # Degenerate fallback: choose the first original sentence.
            scores = np.zeros(len(matrix), dtype=np.float32)
        else:
            centroid /= centroid_norm
            scores = matrix @ centroid

        core_index = int(np.argmax(scores))
        core_vector = matrix[core_index].astype(np.float32)
        core_sentence = sentences[core_index]
        count = len(sentences)

        # Mean pairwise cosine similarity without materializing an n x n matrix:
        # for unit vectors, mean_{i<j}(vi.vj) = (||sum(vi)||^2 - n)/(n(n-1)).
        if count < 2:
            cohesion = 1.0
        else:
            vector_sum = matrix.sum(axis=0, dtype=np.float64)
            cohesion = (
                float(np.dot(vector_sum, vector_sum)) - count
            ) / (count * (count - 1))
            cohesion = float(np.clip(cohesion, -1.0, 1.0))

        result = {
            "playlist_id": playlist_id,
            "playlist_url": playlist_url or playlist_id,
            "title_en": title_en,
            "title_cn": title_cn,
            "intro_en": intro_en,
            "intro_cn": intro_cn,
            "sentence_count": count,
            "cohesion": cohesion,
            "core_sentence_vector": core_vector.tolist(),
            "core_sentence_en": core_sentence,
        }

        sentence_map.clear()  # Explicitly discard vector -> sentence map.
        return result

    def analyze_playlists(
        self,
        playlists: list[dict[str, Any]],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Analyze each playlist sequentially and retain only its core record."""
        self.playlist_cores.clear()
        total = len(playlists)

        for index, playlist in enumerate(playlists, start=1):
            playlist_id = str(playlist["playlist_id"])
            sentences = playlist.get("sentences") or []
            if not sentences:
                continue

            if show_progress:
                print(
                    f"[{index}/{total}] {playlist.get('title', playlist_id)} "
                    f"({len(sentences)} sentences)"
                )

            sentence_map = self.build_sentence_map(
                sentences,
                batch_size=batch_size,
                show_progress=False,
            )
            core = self.find_core_sentence(
                playlist_id=playlist_id,
                title_en=str(playlist.get("title_en", playlist_id)),
                title_cn=str(playlist.get("title_cn", playlist_id)),
                sentence_map=sentence_map,
                playlist_url=playlist.get("playlist_url"),
                intro_en=playlist.get("intro_en"),
                intro_cn=playlist.get("intro_cn"),
            )
            self.playlist_cores[playlist_id] = core

            # find_core_sentence() clears the dictionary; drop the reference too.
            del sentence_map

        return self.playlist_cores

    # ------------------------------------------------------------------
    # Playlist-level hierarchy
    # ------------------------------------------------------------------

    @staticmethod
    def _group_snapshot(
        node_id: str,
        members: list[dict[str, Any]],
        core_matrix: np.ndarray,
    ) -> dict[str, Any]:
        """Describe a cluster side and select its representative playlist core."""
        ids = [member["playlist_id"] for member in members]
        positions = {member["playlist_id"]: i for i, member in enumerate(members)}
        # core_matrix is aligned to the global playlist order; member records carry
        # their own normalized core vector so this method remains self-contained.
        matrix = np.asarray(
            [member["core_sentence_vector"] for member in members], dtype=np.float32
        )
        centroid = matrix.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-12:
            centroid /= norm
            representative_index = int(np.argmax(matrix @ centroid))
        else:
            representative_index = 0
        representative = members[representative_index]

        # playlist_rows = [
        #     {
        #         "playlist_id": member["playlist_id"],
        #         # "title_en": member["title_en"],
        #         # "title_cn": member["title_cn"],
        #         # "intro_en": member["intro_en"],
        #         # "intro_cn": member["intro_cn"],
        #         # "core_sentence_en": member["core_sentence_en"],
        #     }
        #     for member in members
        # ]
        return {
            "node_id": node_id,
            "playlist_count": len(members),
            "playlist_ids": ids,
            # "playlists": playlist_rows,
            # "core_sentences_en": [member["core_sentence_en"] for member in members],
            "representative_playlist_id": representative["playlist_id"],
            # "representative_playlist_title_en": representative["title_en"],
            # "representative_playlist_title_cn": representative["title_cn"],
            # "representative_playlist_intro_en": representative["intro_en"],
            # "representative_playlist_intro_cn": representative["intro_cn"],
            # "representative_core_sentence_en": representative["core_sentence_en"],
        }

    def build_hierarchy(self) -> dict[str, Any]:
        """Average-linkage hierarchical clustering over playlist core vectors.

        Each merge node stores complete left/right member snapshots, including
        each side's playlist names and core sentences. This makes each split
        independently displayable by the frontend without rebuilding the tree.
        """
        records = list(self.playlist_cores.values())
        count = len(records)
        leaves: list[dict[str, Any]] = []
        if count == 0:
            self.hierarchy = {
                "method": "average",
                "metric": "cosine",
                "leaf_count": 0,
                "root_node_id": None,
                "leaves": [],
                "merges": [],
            }
            return self.hierarchy

        for index, record in enumerate(records):
            leaf_id = f"playlist-{index:05d}"
            leaves.append(
                {
                    "node_id": leaf_id,
                    "type": "playlist",
                    "playlist_id": record["playlist_id"],
                    # "title_en": record["title_en"],
                    # "title_cn": record["title_cn"],
                    # "intro_en": record["intro_en"],
                    # "intro_cn": record["intro_cn"],
                    # "core_sentence_en": record["core_sentence_en"],
                }
            )

        if count == 1:
            self.hierarchy = {
                "method": "average",
                "metric": "cosine",
                "leaf_count": 1,
                "root_node_id": leaves[0]["node_id"],
                "leaves": leaves,
                "merges": [],
            }
            return self.hierarchy

        try:
            from scipy.cluster.hierarchy import linkage
            from scipy.spatial.distance import pdist
        except ImportError as exc:
            raise RuntimeError(
                "Hierarchical clustering requires SciPy. Install it with: pip install scipy"
            ) from exc

        matrix = np.asarray(
            [record["core_sentence_vector"] for record in records], dtype=np.float32
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        condensed_distances = pdist(matrix, metric="cosine")
        condensed_distances = np.nan_to_num(
            condensed_distances, nan=1.0, posinf=2.0, neginf=0.0
        )
        linkage_matrix = linkage(condensed_distances, method="average")

        # A group stores ordered playlist records and the id of the tree node.
        groups: dict[int, dict[str, Any]] = {}
        for index, record in enumerate(records):
            groups[index] = {
                "node_id": leaves[index]["node_id"],
                "members": [record],
            }

        merges: list[dict[str, Any]] = []
        for merge_index, row in enumerate(linkage_matrix):
            left_index = int(row[0])
            right_index = int(row[1])
            distance = float(np.clip(row[2], 0.0, 2.0))
            node_number = count + merge_index
            node_id = f"cluster-{merge_index:05d}"

            left_group = groups[left_index]
            right_group = groups[right_index]
            left_members = left_group["members"]
            right_members = right_group["members"]
            merged_members = left_members + right_members

            left_snapshot = self._group_snapshot(
                left_group["node_id"], left_members, matrix
            )
            right_snapshot = self._group_snapshot(
                right_group["node_id"], right_members, matrix
            )
            merged_snapshot = self._group_snapshot(
                node_id, merged_members, matrix
            )

            merges.append(
                {
                    "node_id": node_id,
                    "type": "cluster",
                    "merge_order": merge_index + 1,
                    "distance": distance,
                    "similarity": float(np.clip(1.0 - distance, -1.0, 1.0)),
                    "left": left_snapshot,
                    "right": right_snapshot,
                    "merged": merged_snapshot,
                }
            )
            groups[node_number] = {
                "node_id": node_id,
                "members": merged_members,
            }

        root_node_id = merges[-1]["node_id"]
        self.hierarchy = {
            "method": "average",
            "metric": "cosine",
            "leaf_count": count,
            "root_node_id": root_node_id,
            "leaves": leaves,
            "merges": merges,
        }
        return self.hierarchy

    def build_playlist_neighbors(self, top_k: int = 10) -> dict[str, list[dict[str, Any]]]:
        """Return each playlist's nearest other playlists using core vectors."""
        records = list(self.playlist_cores.values())
        self.neighbors = {}
        if len(records) < 2:
            return self.neighbors

        matrix = np.asarray(
            [record["core_sentence_vector"] for record in records], dtype=np.float32
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        similarities = matrix @ matrix.T
        np.fill_diagonal(similarities, -np.inf)

        k = max(1, min(int(top_k), len(records) - 1))
        for index, record in enumerate(records):
            # Stable tie-break: score descending, then original playlist order.
            order = np.argsort(-similarities[index], kind="stable")[:k]
            self.neighbors[record["playlist_id"]] = [
                {
                    "playlist_id": records[int(other)]["playlist_id"],
                    # "title_en": records[int(other)]["title_en"],
                    # "title_cn": records[int(other)]["title_cn"],
                    # "intro_en": records[int(other)]["intro_en"],
                    # "intro_cn": records[int(other)]["intro_cn"],
                    # "core_sentence_en": records[int(other)]["core_sentence_en"],
                    "similarity": float(similarities[index, int(other)]),
                }
                for other in order
            ]
        return self.neighbors

    # ------------------------------------------------------------------
    # End-to-end analysis and JSON output
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".part")
        with temp.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temp.replace(path)

    def analyze_json(
        self,
        source: str | Path | dict[str, Any],
        output_dir: str | Path | None = "analysis_output",
        batch_size: int = 32,
        top_k: int = 10,
        include_coming_soon: bool = False,
    ) -> dict[str, Any]:
        """Run the complete two-level analysis on the supplied JSON structure."""
        data = self.load_json(source)
        playlists = self.extract_playlists(
            data,
            include_coming_soon=include_coming_soon,
        )
        if not playlists:
            raise ValueError(
                "No playlists with sentences were found. Expected top-level "
                "'contents' mapping whose values contain a 'sentences' array."
            )

        self.analyze_playlists(
            playlists,
            batch_size=batch_size,
            show_progress=True,
        )
        hierarchy = self.build_hierarchy()
        neighbors = self.build_playlist_neighbors(top_k=top_k)
        
        playlist_analysis = {}
        for key in self.playlist_cores:
            pa = {
                "url": self.playlist_cores[key]["playlist_id"],
                "title_en": self.playlist_cores[key]["title_en"],
                "title_cn": self.playlist_cores[key]["title_cn"],
                "intro_en": self.playlist_cores[key]["intro_en"],
                "intro_cn": self.playlist_cores[key]["intro_cn"],
                "core_sentence_en": self.playlist_cores[key]["core_sentence_en"],
            }
        playlist_analysis[key] = pa

        results = {
            "playlist_analysis": playlist_analysis,
            "playlist_hierarchy": hierarchy,
            "playlist_neighbors": neighbors,
        }
        if output_dir is not None:
            self.save_results(output_dir)
        return results

    def save_results(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        playlist_analysis = {}
        for key in self.playlist_cores:
            pa = {
                "url": self.playlist_cores[key]["playlist_id"],
                "title_en": self.playlist_cores[key]["title_en"],
                "title_cn": self.playlist_cores[key]["title_cn"],
                "intro_en": self.playlist_cores[key]["intro_en"],
                "intro_cn": self.playlist_cores[key]["intro_cn"],
                "core_sentence_en": self.playlist_cores[key]["core_sentence_en"],
            }
        playlist_analysis[key] = pa

        self._write_json(
            output_dir / "playlist_analysis.json",
            playlist_analysis,
        )
        if self.hierarchy is None:
            self.build_hierarchy()
        self._write_json(output_dir / "playlist_hierarchy.json", self.hierarchy)
        self._write_json(output_dir / "playlist_neighbors.json", self.neighbors)

        manifest = {
            "format": "innertune-playlist-analysis-v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_repo": self.HF_REPO,
            "model_revision": self.HF_REVISION,
            "model_file": self.model_info["filename"],
            "model_sha256": self.model_info["sha256"],
            "quantization": self.quantization,
            "embedding_dimension": self.EMBEDDING_DIMENSION,
            "input_prefix": self.INPUT_PREFIX,
            "pooling": self.POOLING,
            "normalization": "L2",
            "playlist_core_method": "medoid-nearest-to-normalized-mean",
            "playlist_clustering": "average-linkage/cosine over core-sentence vectors",
            "llama_cpp_version": self.LLAMA_VERSION,
            "playlist_count": len(self.playlist_cores),
            "license": "CC BY-NC 4.0; check terms before commercial use",
        }
        self._write_json(output_dir / "manifest.json", manifest)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
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

    def __enter__(self) -> "PlaylistEmbedding":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed Innertune affirmation playlists and build a hierarchy."
    )
    parser.add_argument("--input", default="innertune_list.json", help="Input JSON path or HTTP(S) URL")
    parser.add_argument("--output", default="analysis_output", help="Output directory for JSON results")
    parser.add_argument("--quantization", default="Q8_0", choices=PlaylistEmbedding.MODEL_FILES)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--no-auto-install-server", action="store_true")
    args = parser.parse_args()

    with PlaylistEmbedding(
        quantization=args.quantization,
        port=args.port,
        threads=args.threads,
        auto_install_server=not args.no_auto_install_server,
    ) as analyzer:
        analyzer.analyze_json(
            args.input,
            output_dir=args.output,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )

    print(f"Analysis complete. JSON files written to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
