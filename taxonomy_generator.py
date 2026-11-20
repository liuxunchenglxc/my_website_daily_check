from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Callable


class TaxonomyGenerationError(RuntimeError):
    """Raised when the hierarchy or generated taxonomy is inconsistent."""


class TaxonomyGenerator:
    """Build and name a compact taxonomy from PlaylistEmbedding's hierarchy.

    Input is the exact result shape returned by PlaylistEmbedding.analyze_json:
        playlist_analysis: {playlist_url: {title_en, title_cn, ...}}
        playlist_hierarchy: {leaves: [...], merges: [...], root_node_id: ...}

    The source average-linkage tree is not modified. It is adaptively cut into:
        root -> generated top categories -> original level-1 clusters
             -> small naming batches -> playlist leaves

    Labels are cached by content fingerprint, not by cluster-NNNNN IDs. This is
    important because merge IDs can change when the playlist order changes.
    """

    PROMPT_VERSION = "taxonomy-qwen35-v1"

    def __init__(
        self,
        *,
        llm: Any,
        translator: Any | None = None,
        cache: dict[str, Any] | None = None,
        translation_cache: dict[str, str] | None = None,
        on_cache_update: Callable[[dict[str, Any]], None] | None = None,
        parent_max_size: int = 56,
        batch_max_size: int = 7,
        top_category_count: int = 8,
        naming_batch_size: int = 16,
        retries: int = 3,
    ) -> None:
        if batch_max_size < 1:
            raise ValueError("batch_max_size must be >= 1")
        if parent_max_size <= batch_max_size:
            raise ValueError("parent_max_size must be greater than batch_max_size")
        if top_category_count < 2:
            raise ValueError("top_category_count must be >= 2")

        self.llm = llm
        self.translator = translator
        self.cache = cache if cache is not None else {}
        self.translation_cache = (
            translation_cache if translation_cache is not None else {}
        )
        self.on_cache_update = on_cache_update
        self.parent_max_size = int(parent_max_size)
        self.batch_max_size = int(batch_max_size)
        self.top_category_count = int(top_category_count)
        self.naming_batch_size = max(1, int(naming_batch_size))
        self.retries = max(1, int(retries))

        self.cache.setdefault("version", 1)
        self.cache.setdefault("labels", {})
        self.cache.setdefault("top_categories", {})

        self._tree: dict[str, dict[str, Any]] = {}
        self._leaves_cache: dict[str, tuple[str, ...]] = {}
        self._analysis_by_playlist: dict[str, dict[str, Any]] = {}
        self._missing_title_count = 0
        self._model_id = str(getattr(llm, "MODEL_ID", "Qwen3.5-2B"))
        self._model_fingerprint = self._hash_payload(
            {
                "model_id": self._model_id,
                "repo": str(getattr(llm, "model_repo", "")),
                "revision": str(getattr(llm, "model_revision", "")),
                "file": str(getattr(llm, "model_file", "")),
            }
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        playlist_analysis: dict[str, dict[str, Any]],
        playlist_hierarchy: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(playlist_analysis, dict) or not playlist_analysis:
            raise TaxonomyGenerationError("playlist_analysis is empty or invalid")
        if not isinstance(playlist_hierarchy, dict):
            raise TaxonomyGenerationError("playlist_hierarchy is invalid")

        self._analysis_by_playlist = self._index_analysis(playlist_analysis)
        self._missing_title_count = 0
        root_id = self._build_tree(playlist_hierarchy)
        root = self._tree[root_id]
        leaf_count = int(root["count"])

        if leaf_count == 0:
            raise TaxonomyGenerationError("The hierarchy contains no leaves")

        # First cut: create broad source clusters while retaining real subtrees.
        level_1_ids = self._cut_by_max_size(root_id, self.parent_max_size)
        level_1_work: list[dict[str, Any]] = []
        batch_work: list[dict[str, Any]] = []

        for parent_id in level_1_ids:
            parent_node = self._tree[parent_id]
            parent_count = int(parent_node["count"])
            terminal = parent_count <= self.batch_max_size

            if terminal:
                child_ids = [parent_id]
            else:
                child_ids = self._cut_by_max_size(
                    parent_id, self.batch_max_size
                )

            level_1_work.append(
                {
                    "node_id": parent_id,
                    "count": parent_count,
                    "terminal": terminal,
                    "child_ids": [] if terminal else child_ids,
                    "fingerprint": self._group_fingerprint(
                        "level1", parent_id
                    ),
                }
            )

            for child_id in child_ids:
                member_leaf_ids = self._get_leaf_ids(child_id)
                members = [self._tree[leaf_id] for leaf_id in member_leaf_ids]
                batch_work.append(
                    {
                        "node_id": child_id,
                        "source_parent_id": parent_id,
                        "count": len(members),
                        "level": 1 if terminal else 2,
                        "member_leaf_ids": list(member_leaf_ids),
                        "members": members,
                        "fingerprint": self._group_fingerprint(
                            "batch", child_id
                        ),
                        "member_uids": [member["uid"] for member in members],
                        "titles_en": [member["title_en"] for member in members],
                    }
                )

        # Stage 1: name the smallest semantic batches.
        self._name_batches(batch_work)

        batch_by_id = {item["node_id"]: item for item in batch_work}

        # Stage 2: name the 35-ish original first-level clusters from their
        # generated child labels. Terminal level-1 clusters reuse their batch.
        self._name_level_1(level_1_work, batch_by_id)

        # Stage 3: make a small number of cross-branch semantic domains.
        categories = self._make_top_categories(level_1_work)

        # Translate generated labels (and direct singleton labels) into Chinese.
        all_named_items: list[dict[str, Any]] = []
        all_named_items.extend(batch_work)
        all_named_items.extend(level_1_work)
        all_named_items.extend(categories)
        self._translate_named_items(all_named_items)

        # Fill terminal level-1 names from their batch entry after translation.
        for level_1_item in level_1_work:
            if level_1_item["terminal"]:
                batch = batch_by_id[level_1_item["node_id"]]
                level_1_item["title_en"] = batch["title_en"]
                level_1_item["title_cn"] = batch["title_cn"]
                level_1_item["label_fingerprint"] = batch["fingerprint"]

        # Map every original level-1 node to exactly one generated domain.
        domain_by_level1: dict[str, str] = {}
        for category in categories:
            for child_id in category["children"]:
                if child_id in domain_by_level1:
                    raise TaxonomyGenerationError(
                        f"Level-1 node assigned more than once: {child_id}"
                    )
                domain_by_level1[child_id] = category["node_id"]

        expected_level1_ids = {item["node_id"] for item in level_1_work}
        if set(domain_by_level1) != expected_level1_ids:
            missing = sorted(expected_level1_ids - set(domain_by_level1))
            extra = sorted(set(domain_by_level1) - expected_level1_ids)
            raise TaxonomyGenerationError(
                f"Top-category coverage mismatch; missing={missing}, extra={extra}"
            )

        level_1_output = []
        for item in level_1_work:
            node = self._tree[item["node_id"]]
            level_1_output.append(
                {
                    "node_id": item["node_id"],
                    "parent_id": domain_by_level1[item["node_id"]],
                    "count": item["count"],
                    "title_en": item["title_en"],
                    "title_cn": item["title_cn"],
                    "terminal": item["terminal"],
                    "children": item["child_ids"],
                    "merge_similarity": node.get("similarity"),
                    "merge_distance": node.get("distance"),
                }
            )

        batch_output = []
        for item in batch_work:
            parent_id = item["source_parent_id"]
            parent_domain_id = domain_by_level1[parent_id]
            output_parent_id = parent_domain_id if item["level"] == 1 else parent_id
            source_node = self._tree[item["node_id"]]
            batch_output.append(
                {
                    "node_id": item["node_id"],
                    "parent_id": output_parent_id,
                    "source_parent_id": parent_id,
                    "level": item["level"],
                    "count": item["count"],
                    "title_en": item["title_en"],
                    "title_cn": item["title_cn"],
                    "uid_list": item["member_uids"],
                    "merge_similarity": source_node.get("similarity"),
                    "merge_distance": source_node.get("distance"),
                }
            )

        categories_output = []
        for category in categories:
            categories_output.append(
                {
                    "node_id": category["node_id"],
                    "title_en": category["title_en"],
                    "title_cn": category["title_cn"],
                    "count": category["count"],
                    "children": category["children"],
                }
            )

        # Cache now contains all generated text; caller may persist it to
        # PersonalStore after generate() returns.
        self._save_cache()

        return {
            "version": "taxonomy-v1",
            "model": self._model_id,
            "model_source": {
                "repo": str(getattr(self.llm, "model_repo", "")),
                "revision": str(getattr(self.llm, "model_revision", "")),
                "file": str(getattr(self.llm, "model_file", "")),
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "parent_max_size": self.parent_max_size,
                "batch_max_size": self.batch_max_size,
                "top_category_count": self.top_category_count,
                "strategy": "adaptive_recursive_cut",
            },
            "stats": {
                "leaf_count": leaf_count,
                "top_category_count": len(categories_output),
                "level_1_count": len(level_1_output),
                "batch_count": len(batch_output),
                "batch_size_min": min(x["count"] for x in batch_output),
                "batch_size_max": max(x["count"] for x in batch_output),
                "batch_size_avg": round(
                    sum(x["count"] for x in batch_output) / len(batch_output), 2
                ),
                "missing_title_en": self._missing_title_count,
            },
            "root": {
                "node_id": root_id,
                "count": leaf_count,
                "children": [item["node_id"] for item in categories_output],
            },
            "categories": categories_output,
            "level_1": level_1_output,
            "batches": batch_output,
        }

    # ------------------------------------------------------------------
    # Original hierarchy reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _index_analysis(
        playlist_analysis: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for key, value in playlist_analysis.items():
            if not isinstance(value, dict):
                continue
            playlist_id = str(value.get("url") or key)
            indexed[playlist_id] = value
        return indexed

    def _build_tree(self, hierarchy: dict[str, Any]) -> str:
        merges_raw = hierarchy.get("merges") or []
        leaves_raw = hierarchy.get("leaves") or []
        root_id = hierarchy.get("root_node_id")

        if not root_id:
            raise TaxonomyGenerationError("playlist_hierarchy.root_node_id is missing")

        merge_by_id = {
            str(item["node_id"]): item
            for item in merges_raw
            if isinstance(item, dict) and item.get("node_id")
        }
        leaf_by_id = {
            str(item["node_id"]): item
            for item in leaves_raw
            if isinstance(item, dict) and item.get("node_id")
        }

        self._tree = {}
        self._leaves_cache = {}
        building: set[str] = set()

        def build_leaf(node_id: str, ref: dict[str, Any] | None = None) -> str:
            ref = ref or {}
            leaf_record = leaf_by_id.get(node_id, {})
            playlist_id = (
                leaf_record.get("playlist_id")
                or ref.get("representative_playlist_id")
            )
            if not playlist_id:
                refs = ref.get("playlist_ids") or []
                if len(refs) == 1:
                    playlist_id = refs[0]
            if not playlist_id:
                raise TaxonomyGenerationError(
                    f"Leaf {node_id!r} has no playlist_id in hierarchy leaves or parent snapshot"
                )

            playlist_id = str(playlist_id)
            analysis = self._analysis_by_playlist.get(playlist_id)
            if analysis is None:
                raise TaxonomyGenerationError(
                    f"No playlist_analysis record for leaf {node_id!r}: {playlist_id}"
                )

            title_en = str(analysis.get("title_en") or "").strip()
            if not title_en:
                self._missing_title_count += 1
                title_en = self._fallback_title(playlist_id)

            uid = hashlib.sha256(playlist_id.encode("utf-8")).hexdigest()
            self._tree[node_id] = {
                "node_id": node_id,
                "kind": "leaf",
                "leaf": True,
                "count": 1,
                "playlist_id": playlist_id,
                "uid": uid,
                "title_en": title_en,
                "title_cn": str(analysis.get("title_cn") or "").strip(),
                "similarity": None,
                "distance": None,
            }
            return node_id

        def build_ref(ref: dict[str, Any]) -> str:
            if not isinstance(ref, dict) or not ref.get("node_id"):
                raise TaxonomyGenerationError(f"Invalid child reference: {ref!r}")
            node_id = str(ref["node_id"])
            if node_id in self._tree:
                return node_id
            if node_id in merge_by_id:
                return build_internal(node_id)
            return build_leaf(node_id, ref)

        def build_internal(node_id: str) -> str:
            if node_id in self._tree:
                return node_id
            if node_id in building:
                raise TaxonomyGenerationError(f"Cycle detected in hierarchy at {node_id}")
            raw = merge_by_id.get(node_id)
            if raw is None:
                raise TaxonomyGenerationError(f"Missing merge record: {node_id}")

            building.add(node_id)
            left_id = build_ref(raw.get("left"))
            right_id = build_ref(raw.get("right"))
            left = self._tree[left_id]
            right = self._tree[right_id]
            count = int(left["count"]) + int(right["count"])

            merged_snapshot = raw.get("merged") or raw.get("merge") or {}
            expected_count = merged_snapshot.get("playlist_count")
            if expected_count is not None and int(expected_count) != count:
                raise TaxonomyGenerationError(
                    f"Merge count mismatch at {node_id}: children={count}, merged={expected_count}"
                )

            self._tree[node_id] = {
                "node_id": node_id,
                "kind": "cluster",
                "leaf": False,
                "count": count,
                "left_id": left_id,
                "right_id": right_id,
                "similarity": raw.get("similarity"),
                "distance": raw.get("distance"),
            }
            building.remove(node_id)
            return node_id

        if str(root_id) in merge_by_id:
            built_root = build_internal(str(root_id))
        elif str(root_id) in leaf_by_id:
            built_root = build_leaf(str(root_id), leaf_by_id[str(root_id)])
        else:
            raise TaxonomyGenerationError(f"Unknown hierarchy root: {root_id}")

        expected_leaf_count = hierarchy.get("leaf_count")
        actual_leaf_count = self._tree[built_root]["count"]
        if expected_leaf_count is not None and int(expected_leaf_count) != actual_leaf_count:
            raise TaxonomyGenerationError(
                f"Hierarchy leaf_count mismatch: metadata={expected_leaf_count}, tree={actual_leaf_count}"
            )

        # Every leaf must be reachable exactly once in a proper binary tree.
        leaf_ids = self._get_leaf_ids(built_root)
        if len(leaf_ids) != actual_leaf_count or len(set(leaf_ids)) != len(leaf_ids):
            raise TaxonomyGenerationError("The rebuilt hierarchy has duplicate or missing leaves")

        declared_leaf_ids = set(leaf_by_id)
        reachable_leaf_ids = set(leaf_ids)
        if declared_leaf_ids and declared_leaf_ids != reachable_leaf_ids:
            missing = sorted(declared_leaf_ids - reachable_leaf_ids)
            undeclared = sorted(reachable_leaf_ids - declared_leaf_ids)
            raise TaxonomyGenerationError(
                "Hierarchy leaf coverage mismatch; "
                f"unreachable={missing[:10]}, undeclared={undeclared[:10]}"
            )
        return built_root

    @staticmethod
    def _fallback_title(playlist_id: str) -> str:
        tail = playlist_id.rstrip("/").split("/")[-1]
        return re.sub(r"[-_]+", " ", tail).strip().title() or "Untitled Playlist"

    def _get_leaf_ids(self, node_id: str) -> tuple[str, ...]:
        cached = self._leaves_cache.get(node_id)
        if cached is not None:
            return cached
        node = self._tree[node_id]
        if node["leaf"]:
            result = (node_id,)
        else:
            result = (
                self._get_leaf_ids(node["left_id"])
                + self._get_leaf_ids(node["right_id"])
            )
        self._leaves_cache[node_id] = result
        return result

    def _cut_by_max_size(self, node_id: str, maximum: int) -> list[str]:
        node = self._tree[node_id]
        if node["leaf"] or int(node["count"]) <= maximum:
            return [node_id]
        return (
            self._cut_by_max_size(node["left_id"], maximum)
            + self._cut_by_max_size(node["right_id"], maximum)
        )

    # ------------------------------------------------------------------
    # Stable content fingerprints and cache
    # ------------------------------------------------------------------

    def _group_fingerprint(
        self,
        kind: str,
        node_id: str,
        extra: list[dict[str, Any]] | None = None,
    ) -> str:
        members = []
        for leaf_id in self._get_leaf_ids(node_id):
            leaf = self._tree[leaf_id]
            members.append(
                {
                    "playlist_id": leaf["playlist_id"],
                    "title_en": leaf["title_en"],
                }
            )
        members.sort(key=lambda item: item["playlist_id"])
        payload = {
            "prompt_version": self.PROMPT_VERSION,
            "model": self._model_fingerprint,
            "kind": kind,
            "members": members,
            "extra": extra or [],
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _cache_label(self, fingerprint: str) -> dict[str, str] | None:
        entry = self.cache.get("labels", {}).get(fingerprint)
        if not isinstance(entry, dict) or not entry.get("title_en"):
            return None
        return {
            "title_en": str(entry["title_en"]).strip(),
            "title_cn": str(entry.get("title_cn") or "").strip(),
        }

    def _store_label(
        self,
        fingerprint: str,
        title_en: str,
        title_cn: str = "",
    ) -> None:
        labels = self.cache.setdefault("labels", {})
        previous = labels.get(fingerprint, {})
        labels[fingerprint] = {
            "title_en": title_en.strip(),
            "title_cn": title_cn.strip() or str(previous.get("title_cn") or ""),
            "model": self._model_id,
            "prompt_version": self.PROMPT_VERSION,
        }

    def _save_cache(self) -> None:
        if self.on_cache_update is not None:
            self.on_cache_update(self.cache)

    # ------------------------------------------------------------------
    # LLM label generation
    # ------------------------------------------------------------------

    @staticmethod
    def _string_mapping_schema(keys: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                key: {"type": "string", "minLength": 1, "maxLength": 80}
                for key in keys
            },
            "required": keys,
            "additionalProperties": False,
        }

    def _generate_label_mapping(
        self,
        groups: list[dict[str, Any]],
        *,
        schema_name: str,
        task_description: str,
    ) -> dict[str, str]:
        ids = [str(group["node_id"]) for group in groups]
        schema = self._string_mapping_schema(ids)
        prompt_payload = {
            "groups": [
                {
                    "node_id": group["node_id"],
                    "count": group.get("count", len(group.get("titles_en", []))),
                    "titles_en": group.get("titles_en", []),
                    "child_categories": group.get("child_categories", []),
                }
                for group in groups
            ]
        }
        user_prompt = (
            f"{task_description}\n\n"
            "Return one concise English taxonomy label for every node_id. "
            "Use 2-6 words, natural title case, no numbering, no explanation. "
            "Do not invent topics that are absent from the input. If a group is "
            "slightly heterogeneous, choose the narrowest honest umbrella label.\n\n"
            "Input JSON:\n"
            + json.dumps(prompt_payload, ensure_ascii=False, indent=2)
        )
        system_prompt = (
            "You are a careful taxonomy editor for an affirmation and self-development "
            "library. Your job is to name semantic groups, not to rewrite their source "
            "titles. Keep labels short, specific, and non-overlapping where possible. "
            "Output only the JSON object required by the schema."
        )

        last_error = ""
        for attempt in range(1, self.retries + 1):
            prompt = user_prompt
            if last_error:
                prompt += (
                    "\n\nYour previous output failed validation: "
                    + last_error
                    + "\nRegenerate the complete JSON object."
                )
            try:
                result = self.llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    schema=schema,
                    schema_name=schema_name,
                    max_tokens=max(512, len(groups) * 80),
                )
            except Exception as exc:
                last_error = f"model request failed: {type(exc).__name__}: {exc}"
                continue
            if not isinstance(result, dict):
                last_error = "output is not an object"
                continue
            missing = sorted(set(ids) - set(result))
            unknown = sorted(set(result) - set(ids))
            values_ok = all(
                isinstance(result.get(node_id), str)
                and bool(result[node_id].strip())
                and len(result[node_id].strip()) <= 80
                for node_id in ids
            )
            if not missing and not unknown and values_ok:
                return {node_id: result[node_id].strip() for node_id in ids}
            last_error = (
                f"missing keys={missing}; unknown keys={unknown}; "
                f"all values must be non-empty strings <= 80 chars"
            )

        raise TaxonomyGenerationError(
            f"LLM failed to generate valid labels for {schema_name}: {last_error}"
        )

    def _name_batches(self, batches: list[dict[str, Any]]) -> None:
        pending: list[dict[str, Any]] = []
        for batch in batches:
            cached = self._cache_label(batch["fingerprint"])
            if cached:
                batch.update(cached)
                continue

            # A singleton has no theme to abstract: reuse its original playlist title.
            if batch["count"] == 1:
                batch["title_en"] = batch["titles_en"][0]
                leaf = batch["members"][0]
                batch["title_cn"] = leaf.get("title_cn", "")
                self._store_label(
                    batch["fingerprint"],
                    batch["title_en"],
                    batch["title_cn"],
                )
                continue

            pending.append(batch)

        # Persist singleton labels before invoking the model.
        self._save_cache()

        for offset in range(0, len(pending), self.naming_batch_size):
            chunk = pending[offset : offset + self.naming_batch_size]
            mapping = self._generate_label_mapping(
                chunk,
                schema_name="taxonomy_batch_labels",
                task_description=(
                    "Name each small cluster from its original English playlist titles. "
                    "The cluster is one semantic subtree. Do not return individual "
                    "playlist titles as a list; return one umbrella category name."
                ),
            )
            for item in chunk:
                item["title_en"] = mapping[item["node_id"]]
                item["title_cn"] = ""
                self._store_label(item["fingerprint"], item["title_en"])
            self._save_cache()

    def _name_level_1(
        self,
        level_1: list[dict[str, Any]],
        batch_by_id: dict[str, dict[str, Any]],
    ) -> None:
        pending: list[dict[str, Any]] = []

        for parent in level_1:
            if parent["terminal"]:
                batch = batch_by_id[parent["node_id"]]
                parent["title_en"] = batch["title_en"]
                parent["title_cn"] = batch.get("title_cn", "")
                parent["fingerprint"] = batch["fingerprint"]
                continue

            child_categories = [
                {
                    "title_en": batch_by_id[child_id]["title_en"],
                    "count": batch_by_id[child_id]["count"],
                }
                for child_id in parent["child_ids"]
            ]
            parent["child_categories"] = child_categories
            parent["fingerprint"] = self._group_fingerprint(
                "level1",
                parent["node_id"],
                extra=child_categories,
            )

            cached = self._cache_label(parent["fingerprint"])
            if cached:
                parent.update(cached)
            else:
                pending.append(parent)

        for offset in range(0, len(pending), self.naming_batch_size):
            chunk = pending[offset : offset + self.naming_batch_size]
            mapping = self._generate_label_mapping(
                chunk,
                schema_name="taxonomy_level1_labels",
                task_description=(
                    "Name each broader topic by summarizing its child-category labels. "
                    "Treat the child labels as the evidence for the broader theme. "
                    "Do not simply concatenate child labels."
                ),
            )
            for item in chunk:
                item["title_en"] = mapping[item["node_id"]]
                item["title_cn"] = ""
                self._store_label(item["fingerprint"], item["title_en"])
            self._save_cache()

    # ------------------------------------------------------------------
    # Cross-branch top-level domains
    # ------------------------------------------------------------------

    def _make_top_categories(
        self, level_1: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if len(level_1) < self.top_category_count:
            raise TaxonomyGenerationError(
                f"Need at least {self.top_category_count} level-1 topics, "
                f"got {len(level_1)}"
            )

        source = [
            {
                "node_id": item["node_id"],
                "fingerprint": item["fingerprint"],
                "title_en": item["title_en"],
                "count": item["count"],
            }
            for item in level_1
        ]
        source.sort(key=lambda row: row["node_id"])
        prompt_source = [
            {
                "node_id": item["node_id"],
                "title_en": item["title_en"],
                "count": item["count"],
            }
            for item in source
        ]
        top_fingerprint = self._hash_payload(
            {
                "prompt_version": self.PROMPT_VERSION,
                "model": self._model_fingerprint,
                "top_category_count": self.top_category_count,
                "source": source,
            }
        )

        cached = self.cache.get("top_categories", {}).get(top_fingerprint)
        if isinstance(cached, list) and cached:
            current_by_fp = {item["fingerprint"]: item for item in level_1}
            categories = []
            cache_valid = True
            for saved in cached:
                member_fps = saved.get("member_fingerprints") or []
                members = [current_by_fp.get(fp) for fp in member_fps]
                if not members or any(member is None for member in members):
                    cache_valid = False
                    break
                categories.append(
                    self._make_category(
                        str(saved["title_en"]),
                        members,  # type: ignore[arg-type]
                        str(saved.get("title_cn") or ""),
                    )
                )
            if cache_valid and self._valid_category_coverage(categories, level_1):
                return categories

        known_ids = [item["node_id"] for item in level_1]
        category_schema = {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "minItems": self.top_category_count,
                    "maxItems": self.top_category_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title_en": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 80,
                            },
                            "member_node_ids": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": len(level_1),
                                "uniqueItems": True,
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["title_en", "member_node_ids"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["categories"],
            "additionalProperties": False,
        }

        user_prompt = (
            f"Group the following {len(level_1)} existing topics into exactly "
            f"{self.top_category_count} broad, user-facing top-level categories.\n"
            "Every input node_id must appear exactly once across member_node_ids. "
            "Do not omit, duplicate, rename, or invent node IDs. A category may "
            "combine topics from different branches of the source hierarchy. "
            "Prefer clear everyday concepts over vague labels such as General or Other. "
            "Keep category boundaries semantically coherent; balance sizes only "
            "when it does not damage semantic coherence. Return English names only.\n\n"
            "Topics:\n"
            + json.dumps(prompt_source, ensure_ascii=False, indent=2)
            + "\n\nAllowed node IDs:\n"
            + json.dumps(known_ids, ensure_ascii=False)
        )
        system_prompt = (
            "You are an expert information architect. Organize a collection of "
            "self-development and affirmation topics into a compact, intuitive "
            "taxonomy. Do not force unrelated subjects together merely to balance "
            "counts. Output only the JSON object required by the schema."
        )

        last_error = ""
        raw_categories: list[dict[str, Any]] | None = None
        for attempt in range(1, self.retries + 1):
            prompt = user_prompt
            if last_error:
                prompt += (
                    "\n\nYour previous output failed validation: "
                    + last_error
                    + "\nRegenerate all categories and ensure exact coverage."
                )
            try:
                result = self.llm.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=prompt,
                    schema=category_schema,
                    schema_name="taxonomy_top_categories",
                    max_tokens=2048,
                )
            except Exception as exc:
                last_error = f"model request failed: {type(exc).__name__}: {exc}"
                continue
            candidate = result.get("categories") if isinstance(result, dict) else None
            if not isinstance(candidate, list) or len(candidate) != self.top_category_count:
                last_error = f"expected exactly {self.top_category_count} categories"
                continue

            assigned: list[str] = []
            names: list[str] = []
            malformed = False
            for category in candidate:
                if not isinstance(category, dict):
                    malformed = True
                    break
                name = category.get("title_en")
                members = category.get("member_node_ids")
                if not isinstance(name, str) or not name.strip():
                    malformed = True
                    break
                if not isinstance(members, list) or not members:
                    malformed = True
                    break
                if any(not isinstance(value, str) for value in members):
                    malformed = True
                    break
                names.append(name.strip())
                assigned.extend(members)

            if malformed:
                last_error = "category objects contain invalid names or member_node_ids"
                continue

            unknown = sorted(set(assigned) - set(known_ids))
            missing = sorted(set(known_ids) - set(assigned))
            duplicates = sorted(
                node_id for node_id in set(assigned) if assigned.count(node_id) > 1
            )
            duplicate_names = len({name.casefold() for name in names}) != len(names)
            if unknown or missing or duplicates or duplicate_names:
                last_error = (
                    f"unknown={unknown}; missing={missing}; duplicates={duplicates}; "
                    f"duplicate category names={duplicate_names}"
                )
                continue

            raw_categories = candidate
            break

        if raw_categories is None:
            raise TaxonomyGenerationError(
                f"Failed to generate top-level categories: {last_error}"
            )

        level1_by_id = {item["node_id"]: item for item in level_1}
        categories = []
        for item in raw_categories:
            members = [level1_by_id[node_id] for node_id in item["member_node_ids"]]
            categories.append(
                self._make_category(item["title_en"].strip(), members)
            )

        if not self._valid_category_coverage(categories, level_1):
            raise TaxonomyGenerationError("Top-level category assignment is incomplete")

        # Cache by level-1 content fingerprints rather than ephemeral cluster IDs.
        cache_value = []
        for category in categories:
            cache_value.append(
                {
                    "title_en": category["title_en"],
                    "title_cn": "",
                    "member_fingerprints": sorted(
                        member["fingerprint"] for member in category["members"]
                    ),
                }
            )
        self.cache.setdefault("top_categories", {})[top_fingerprint] = cache_value
        self._save_cache()
        return categories

    @staticmethod
    def _make_category(
        title_en: str,
        members: list[dict[str, Any]],
        title_cn: str = "",
    ) -> dict[str, Any]:
        member_ids = sorted(member["node_id"] for member in members)
        member_fingerprints = sorted(member["fingerprint"] for member in members)
        membership_fingerprint = TaxonomyGenerator._hash_payload(
            {"kind": "domain-members-v1", "members": member_fingerprints}
        )
        fingerprint = TaxonomyGenerator._hash_payload(
            {"members": member_fingerprints, "title_en": title_en}
        )
        node_id = "domain-" + membership_fingerprint[:12]
        return {
            "node_id": node_id,
            "title_en": title_en,
            "title_cn": title_cn,
            "count": sum(int(member["count"]) for member in members),
            "children": member_ids,
            "members": members,
            "fingerprint": fingerprint,
        }

    @staticmethod
    def _valid_category_coverage(
        categories: list[dict[str, Any]], level_1: list[dict[str, Any]]
    ) -> bool:
        assigned = [child for category in categories for child in category["children"]]
        expected = [item["node_id"] for item in level_1]
        return len(assigned) == len(set(assigned)) and set(assigned) == set(expected)

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def _translate_named_items(self, items: list[dict[str, Any]]) -> None:
        # First collect unique non-empty English labels.
        unique: list[str] = []
        seen: set[str] = set()
        for item in items:
            title_en = str(item.get("title_en") or "").strip()
            if title_en and title_en not in seen:
                seen.add(title_en)
                unique.append(title_en)

        translated: dict[str, str] = {}
        missing: list[str] = []
        # Preserve source Chinese titles for singleton batches. Generated group
        # labels have an empty title_cn and are translated through Hy-MT2.
        for item in items:
            title_en = str(item.get("title_en") or "").strip()
            title_cn = str(item.get("title_cn") or "").strip()
            if title_en and title_cn:
                translated.setdefault(title_en, title_cn)
                self.translation_cache.setdefault(title_en, title_cn)

        for title_en in unique:
            cached = str(self.translation_cache.get(title_en) or "").strip()
            if cached:
                translated[title_en] = cached
            elif title_en not in translated:
                missing.append(title_en)

        if missing and self.translator is not None:
            for index, title_en in enumerate(missing, 1):
                chinese = self.translator.translate(
                    title_en,
                    target="Simplified Chinese",
                    temperature=0.2,
                    max_tokens=128,
                ).strip()
                if not chinese:
                    raise TaxonomyGenerationError(
                        f"Hy-MT2 returned an empty translation for {title_en!r}"
                    )
                self.translation_cache[title_en] = chinese
                translated[title_en] = chinese
                if index % 10 == 0:
                    self._save_cache()
                print(f"Taxonomy translated: {index}/{len(missing)}")

        elif missing:
            # Standalone fallback: use Qwen itself in structured translation batches.
            for offset in range(0, len(missing), 20):
                chunk = missing[offset : offset + 20]
                keys = [f"t{i:03d}" for i in range(len(chunk))]
                schema = self._string_mapping_schema(keys)
                payload = {key: text for key, text in zip(keys, chunk)}
                result = self.llm.generate_json(
                    system_prompt=(
                        "You are a professional English-to-Simplified-Chinese translator. "
                        "Translate short taxonomy labels faithfully and concisely. "
                        "Return only the JSON object required by the schema."
                    ),
                    user_prompt=(
                        "Translate each English label into a natural Simplified Chinese "
                        "category name. Preserve its meaning; do not add explanations.\n\n"
                        + json.dumps(payload, ensure_ascii=False, indent=2)
                    ),
                    schema=schema,
                    schema_name="taxonomy_chinese_translations",
                    max_tokens=1024,
                )
                for key, title_en in zip(keys, chunk):
                    chinese = str(result.get(key) or "").strip()
                    if not chinese:
                        raise TaxonomyGenerationError(
                            f"Qwen3.5 returned an empty Chinese translation for {title_en!r}"
                        )
                    self.translation_cache[title_en] = chinese
                    translated[title_en] = chinese
                self._save_cache()

        for item in items:
            title_en = str(item.get("title_en") or "").strip()
            item["title_cn"] = translated.get(title_en, "")
            fingerprint = item.get("fingerprint")
            if fingerprint and title_en:
                self._store_label(
                    str(fingerprint), title_en, item["title_cn"]
                )

        # Keep top-domain translations with the domain-set cache as well.
        for cached_groups in self.cache.get("top_categories", {}).values():
            for category in cached_groups:
                title_en = str(category.get("title_en") or "").strip()
                if title_en in translated:
                    category["title_cn"] = translated[title_en]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_payload(payload: Any) -> str:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
