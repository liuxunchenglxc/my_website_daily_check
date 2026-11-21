from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Callable, Iterable


class TaxonomyGenerator:
    """Build a compact, named multi-level taxonomy from PlaylistEmbedding output.

    Qwen only receives plain-text title groups and returns plain-text numbered
    lines. This class owns all data identity, tree traversal, parent-child
    relationships, validation, Chinese translation, caching, and JSON output.
    """

    CACHE_SCHEMA = 2
    PROMPT_VERSION = "taxonomy-plain-text-v2"

    def __init__(
        self,
        llm,
        translator,
        cache: dict | None = None,
        translation_cache: dict | None = None,
        on_cache_update: Callable[[dict], None] | None = None,
        parent_max_size: int = 56,
        batch_max_size: int = 7,
        top_category_count: int = 8,
        naming_batch_size: int = 16,
        assignment_batch_size: int = 12,
        persist_translation_every: int = 16,
    ):
        self.llm = llm
        self.translator = translator
        self.cache = cache if isinstance(cache, dict) else {}
        self.translation_cache = (
            translation_cache if isinstance(translation_cache, dict) else {}
        )
        self.on_cache_update = on_cache_update
        self.parent_max_size = max(2, int(parent_max_size))
        self.batch_max_size = max(1, int(batch_max_size))
        if self.batch_max_size >= self.parent_max_size:
            raise ValueError("batch_max_size must be smaller than parent_max_size")
        self.top_category_count = max(1, int(top_category_count))
        self.naming_batch_size = max(1, int(naming_batch_size))
        self.assignment_batch_size = max(1, int(assignment_batch_size))
        self.persist_translation_every = max(1, int(persist_translation_every))

        # A previous JSON-output prototype used a different cache schema.
        # Do not reinterpret its contents as plain-text label fingerprints.
        if (
            self.cache.get("schema_version") != self.CACHE_SCHEMA
            or self.cache.get("prompt_version") != self.PROMPT_VERSION
        ):
            self.cache.clear()
            self.cache.update({
                "schema_version": self.CACHE_SCHEMA,
                "prompt_version": self.PROMPT_VERSION,
                "batch_labels": {},
                "level1_labels": {},
                "topology": {},
            })
        for key in ("batch_labels", "level1_labels", "topology"):
            if not isinstance(self.cache.get(key), dict):
                self.cache[key] = {}

    # ------------------------------------------------------------------
    # Stable fingerprints and persistence
    # ------------------------------------------------------------------
    @staticmethod
    def _hash(payload: Any) -> str:
        raw = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _fingerprint(self, kind: str, payload: Any) -> str:
        return self._hash({
            "schema": self.CACHE_SCHEMA,
            "prompt": self.PROMPT_VERSION,
            "kind": kind,
            "payload": payload,
        })

    def _persist(self) -> None:
        if self.on_cache_update:
            self.on_cache_update(self.cache)

    @staticmethod
    def _chunks(items: list, size: int):
        for start in range(0, len(items), size):
            yield items[start:start + size]

    # ------------------------------------------------------------------
    # Rebuild the exact binary hierarchy produced by PlaylistEmbedding
    # ------------------------------------------------------------------
    def _build_tree(self, playlist_analysis: dict, hierarchy: dict):
        if not isinstance(playlist_analysis, dict) or not playlist_analysis:
            raise ValueError("playlist_analysis is empty or invalid")
        if not isinstance(hierarchy, dict):
            raise ValueError("playlist_hierarchy is invalid")

        leaves_raw = hierarchy.get("leaves") or []
        merges_raw = hierarchy.get("merges") or []
        root_id = hierarchy.get("root_node_id")

        leaf_records: dict[str, dict] = {}
        for item in leaves_raw:
            node_id = item.get("node_id")
            playlist_id = item.get("playlist_id")
            if not node_id or not playlist_id:
                raise ValueError(f"Invalid hierarchy leaf: {item!r}")
            if node_id in leaf_records:
                raise ValueError(f"Duplicate leaf node_id: {node_id}")
            if playlist_id not in playlist_analysis:
                raise ValueError(
                    f"Leaf playlist_id missing from playlist_analysis: {playlist_id}"
                )
            record = playlist_analysis[playlist_id]
            title_en = str(record.get("title_en") or "").strip()
            if not title_en:
                raise ValueError(f"Missing title_en for playlist: {playlist_id}")
            leaf_records[node_id] = {
                "node_id": node_id,
                "playlist_id": playlist_id,
                "uid": hashlib.sha256(playlist_id.encode("utf-8")).hexdigest(),
                "title_en": title_en,
                "title_cn": str(record.get("title_cn") or "").strip(),
            }

        merge_records: dict[str, dict] = {}
        for item in merges_raw:
            node_id = item.get("node_id")
            if not node_id:
                raise ValueError(f"Merge record missing node_id: {item!r}")
            if node_id in merge_records:
                raise ValueError(f"Duplicate merge node_id: {node_id}")
            merge_records[node_id] = item

        if not root_id:
            if len(leaf_records) == 1 and not merge_records:
                root_id = next(iter(leaf_records))
            else:
                raise ValueError("playlist_hierarchy has no root_node_id")

        nodes: dict[str, dict] = {}
        visiting: set[str] = set()

        def visit(node_id: str) -> str:
            if node_id in nodes:
                return node_id
            if node_id in visiting:
                raise ValueError(f"Cycle detected in hierarchy at {node_id}")
            visiting.add(node_id)

            if node_id in leaf_records:
                leaf = leaf_records[node_id]
                nodes[node_id] = {
                    **leaf,
                    "count": 1,
                    "leaf": True,
                    "left_id": None,
                    "right_id": None,
                    "similarity": None,
                    "distance": None,
                }
            else:
                merge = merge_records.get(node_id)
                if merge is None:
                    raise ValueError(
                        f"Node '{node_id}' is neither a hierarchy leaf nor a merge"
                    )
                left = (merge.get("left") or {}).get("node_id")
                right = (merge.get("right") or {}).get("node_id")
                if not left or not right:
                    raise ValueError(f"Merge '{node_id}' has invalid children")
                visit(left)
                visit(right)
                nodes[node_id] = {
                    "node_id": node_id,
                    "count": nodes[left]["count"] + nodes[right]["count"],
                    "leaf": False,
                    "left_id": left,
                    "right_id": right,
                    "similarity": merge.get("similarity"),
                    "distance": merge.get("distance"),
                }
            visiting.remove(node_id)
            return node_id

        visit(root_id)
        root_count = nodes[root_id]["count"]
        expected_count = int(hierarchy.get("leaf_count") or len(leaf_records))
        if root_count != expected_count:
            raise ValueError(
                f"Hierarchy root count mismatch: built={root_count}, "
                f"leaf_count={expected_count}"
            )
        if root_count != len(leaf_records):
            raise ValueError(
                f"Unreachable leaves detected: root={root_count}, "
                f"leaves={len(leaf_records)}"
            )
        if root_count != len(playlist_analysis):
            missing = set(playlist_analysis) - {
                record["playlist_id"] for record in leaf_records.values()
            }
            extra = {
                record["playlist_id"] for record in leaf_records.values()
            } - set(playlist_analysis)
            raise ValueError(
                "Hierarchy/analysis coverage mismatch: "
                f"analysis={len(playlist_analysis)}, leaves={len(leaf_records)}, "
                f"missing={len(missing)}, extra={len(extra)}"
            )

        return root_id, nodes, leaf_records

    @staticmethod
    def _cut_by_max_size(node_id: str, nodes: dict[str, dict], max_size: int) -> list[str]:
        node = nodes[node_id]
        if node["leaf"] or node["count"] <= max_size:
            return [node_id]
        left_groups = TaxonomyGenerator._cut_by_max_size(
            node["left_id"], nodes, max_size
        )
        right_groups = TaxonomyGenerator._cut_by_max_size(
            node["right_id"], nodes, max_size
        )
        return left_groups + right_groups

    @staticmethod
    def _collect_leaves(node_id: str, nodes: dict[str, dict], memo: dict[str, tuple[str, ...]]):
        if node_id in memo:
            return memo[node_id]
        node = nodes[node_id]
        if node["leaf"]:
            result = (node_id,)
        else:
            result = (
                TaxonomyGenerator._collect_leaves(node["left_id"], nodes, memo)
                + TaxonomyGenerator._collect_leaves(node["right_id"], nodes, memo)
            )
        memo[node_id] = result
        return result

    # ------------------------------------------------------------------
    # Qwen naming: only titles in, plain text labels out
    # ------------------------------------------------------------------
    def _name_batch_nodes(self, batch_specs: list[dict]) -> None:
        labels = self.cache["batch_labels"]
        pending = [spec for spec in batch_specs if spec["fingerprint"] not in labels]
        for chunk in self._chunks(pending, self.naming_batch_size):
            groups = [spec["titles_en"] for spec in chunk]
            names = self.llm.name_groups(groups, level="fine")
            if len(names) != len(chunk):
                raise RuntimeError(
                    f"Qwen fine-label count mismatch: expected {len(chunk)}, got {len(names)}"
                )
            for spec, name in zip(chunk, names):
                name = str(name).strip()
                if not name:
                    raise RuntimeError(f"Qwen returned an empty label for {spec['node_id']}")
                labels[spec["fingerprint"]] = {"title_en": name}
            self._persist()
            print(f"[Taxonomy] Fine labels: {len(labels)} cached")

    def _name_level1_nodes(self, level1_specs: list[dict]) -> None:
        labels = self.cache["level1_labels"]
        pending = [
            spec for spec in level1_specs
            if not spec["terminal"] and spec["fingerprint"] not in labels
        ]
        for chunk in self._chunks(pending, self.naming_batch_size):
            groups = [spec["child_titles"] for spec in chunk]
            names = self.llm.name_groups(groups, level="parent")
            if len(names) != len(chunk):
                raise RuntimeError(
                    f"Qwen parent-label count mismatch: expected {len(chunk)}, got {len(names)}"
                )
            for spec, name in zip(chunk, names):
                name = str(name).strip()
                if not name:
                    raise RuntimeError(f"Qwen returned an empty parent label for {spec['node_id']}")
                labels[spec["fingerprint"]] = {"title_en": name}
            self._persist()
            print(f"[Taxonomy] Parent labels: {len(labels)} cached")

    def _build_top_categories(self, level1_specs: list[dict]):
        topics = [
            (spec["title_en"], spec["count"])
            for spec in level1_specs
        ]
        topic_fingerprints = [spec["fingerprint"] for spec in level1_specs]
        topology_fingerprint = self._fingerprint(
            "topology",
            sorted((spec["fingerprint"], spec["title_en"], spec["count"]) for spec in level1_specs),
        )
        saved = self.cache.get("topology", {})
        if saved.get("fingerprint") == topology_fingerprint:
            categories = saved.get("categories") or []
            assignments_by_fp = saved.get("assignments") or {}
            if (
                categories
                and 1 <= len(categories) <= min(self.top_category_count, len(level1_specs))
                and all(fp in assignments_by_fp for fp in topic_fingerprints)
            ):
                assignments = [int(assignments_by_fp[fp]) for fp in topic_fingerprints]
                if all(1 <= value <= len(categories) for value in assignments):
                    print("[Taxonomy] Reusing cached top-level taxonomy")
                    return categories, assignments

        category_count = min(self.top_category_count, len(level1_specs))
        category_titles = self.llm.name_categories(topics, category_count)
        if len(category_titles) != category_count:
            raise RuntimeError(
                f"Qwen top-category count mismatch: expected {category_count}, "
                f"got {len(category_titles)}"
            )
        assignments = self.llm.assign_categories(
            [spec["title_en"] for spec in level1_specs],
            category_titles,
            batch_size=self.assignment_batch_size,
        )
        if len(assignments) != len(level1_specs):
            raise RuntimeError("Qwen category assignment count mismatch")
        if any(not isinstance(value, int) or value < 1 or value > category_count for value in assignments):
            raise RuntimeError("Qwen returned an invalid category index")

        used = set(assignments)
        # Remove unused category labels instead of creating empty map regions.
        if len(used) < category_count:
            remap = {old: new for new, old in enumerate(sorted(used), 1)}
            category_titles = [category_titles[old - 1] for old in sorted(used)]
            assignments = [remap[value] for value in assignments]

        self.cache["topology"] = {
            "fingerprint": topology_fingerprint,
            "categories": category_titles,
            "assignments": {
                fp: assignment
                for fp, assignment in zip(topic_fingerprints, assignments)
            },
        }
        self._persist()
        return category_titles, assignments

    # ------------------------------------------------------------------
    # Translation and output assembly
    # ------------------------------------------------------------------
    def _translate_labels(self, labels: list[str]) -> dict[str, str]:
        result: dict[str, str] = {}
        new_count = 0
        for title_en in dict.fromkeys(str(x).strip() for x in labels if str(x).strip()):
            title_cn = self.translation_cache.get(title_en)
            if not title_cn:
                if self.translator is None:
                    raise RuntimeError("translator is required to generate title_cn")
                title_cn = self.translator.translate(
                    title_en,
                    target="Simplified Chinese",
                    temperature=0.2,
                    max_tokens=128,
                ).strip()
                if not title_cn:
                    raise RuntimeError(f"Translator returned empty text for: {title_en}")
                self.translation_cache[title_en] = title_cn
                new_count += 1
                if new_count % self.persist_translation_every == 0:
                    self._persist()
            result[title_en] = title_cn
        self._persist()
        return result

    @staticmethod
    def _node_metadata(node_id: str, nodes: dict[str, dict]) -> dict:
        node = nodes[node_id]
        return {
            "node_id": node_id,
            "count": node["count"],
            "merge_similarity": node.get("similarity"),
            "merge_distance": node.get("distance"),
            "leaf": node["leaf"],
        }

    def generate(self, playlist_analysis: dict, playlist_hierarchy: dict) -> dict:
        root_id, nodes, leaf_records = self._build_tree(
            playlist_analysis, playlist_hierarchy
        )
        leaf_memo: dict[str, tuple[str, ...]] = {}

        level1_ids = self._cut_by_max_size(
            root_id, nodes, self.parent_max_size
        )
        batch_specs: list[dict] = []
        level1_specs: list[dict] = []

        for parent_id in level1_ids:
            parent_leaf_ids = self._collect_leaves(parent_id, nodes, leaf_memo)
            parent_playlist_ids = [leaf_records[x]["playlist_id"] for x in parent_leaf_ids]
            parent_count = len(parent_leaf_ids)

            if parent_count <= self.batch_max_size:
                batch_node_ids = [parent_id]
                terminal = True
            else:
                batch_node_ids = self._cut_by_max_size(
                    parent_id, nodes, self.batch_max_size
                )
                terminal = False

            child_batch_specs = []
            for batch_node_id in batch_node_ids:
                batch_leaf_ids = self._collect_leaves(batch_node_id, nodes, leaf_memo)
                batch_records = [leaf_records[x] for x in batch_leaf_ids]
                titles_en = [item["title_en"] for item in batch_records]
                members_for_fingerprint = sorted(
                    (item["playlist_id"], item["title_en"])
                    for item in batch_records
                )
                fingerprint = self._fingerprint("fine-batch", members_for_fingerprint)
                spec = {
                    "node_id": batch_node_id,
                    "parent_id": parent_id,
                    "fingerprint": fingerprint,
                    "count": len(batch_records),
                    "titles_en": titles_en,
                    "leaf_ids": list(batch_leaf_ids),
                    "playlist_ids": [item["playlist_id"] for item in batch_records],
                    "uids": [item["uid"] for item in batch_records],
                }
                batch_specs.append(spec)
                child_batch_specs.append(spec)

            if terminal:
                parent_fp = child_batch_specs[0]["fingerprint"]
                child_titles = [
                    self.cache.get("batch_labels", {}).get(parent_fp, {}).get("title_en")
                ]
            else:
                child_titles = []

            level1_fingerprint = self._fingerprint(
                "level1",
                {
                    "members": sorted(parent_playlist_ids),
                    "children": [spec["fingerprint"] for spec in child_batch_specs],
                },
            )
            level1_specs.append({
                "node_id": parent_id,
                "fingerprint": level1_fingerprint,
                "count": parent_count,
                "terminal": terminal,
                "children": child_batch_specs,
                "child_titles": child_titles,
                "playlist_ids": parent_playlist_ids,
                "leaf_ids": list(parent_leaf_ids),
                "uids": [leaf_records[x]["uid"] for x in parent_leaf_ids],
            })

        # Name all fine-grained batches first.
        self._name_batch_nodes(batch_specs)

        # Fill child titles now that all fine labels are available.
        for spec in level1_specs:
            if spec["terminal"]:
                only_batch = spec["children"][0]
                spec["title_en"] = self.cache["batch_labels"][only_batch["fingerprint"]]["title_en"]
                spec["fingerprint"] = self._fingerprint(
                    "level1-terminal",
                    {"members": sorted(spec["playlist_ids"]), "title": spec["title_en"]},
                )
            else:
                spec["child_titles"] = [
                    self.cache["batch_labels"][child["fingerprint"]]["title_en"]
                    for child in spec["children"]
                ]

        self._name_level1_nodes(level1_specs)
        for spec in level1_specs:
            if not spec["terminal"]:
                spec["title_en"] = self.cache["level1_labels"][spec["fingerprint"]]["title_en"]

        # Create/reuse the top level and assign every level-1 theme exactly once.
        category_titles, assignments = self._build_top_categories(level1_specs)

        all_titles = list(category_titles)
        all_titles.extend(spec["title_en"] for spec in level1_specs)
        all_titles.extend(
            self.cache["batch_labels"][spec["fingerprint"]]["title_en"]
            for spec in batch_specs
        )
        translations = self._translate_labels(all_titles)

        categories = [
            {
                "id": f"domain-{index:02d}",
                "title_en": title,
                "title_cn": translations[title],
                "count": 0,
                "children": [],
            }
            for index, title in enumerate(category_titles, 1)
        ]

        for spec, category_index in zip(level1_specs, assignments):
            title_en = spec["title_en"]
            level1_node = {
                "id": f"topic:{spec['node_id']}",
                **self._node_metadata(spec["node_id"], nodes),
                "title_en": title_en,
                "title_cn": translations[title_en],
                "parent_id": f"domain-{category_index:02d}",
                "children": [],
            }

            if spec["terminal"]:
                leaf_specs = spec["children"][0]
                leaves = [
                    {
                        "id": f"leaf:{leaf_records[leaf_id]['uid']}",
                        "node_id": leaf_id,
                        "playlist_id": leaf_records[leaf_id]["playlist_id"],
                        "uid": leaf_records[leaf_id]["uid"],
                        "title_en": leaf_records[leaf_id]["title_en"],
                        "title_cn": leaf_records[leaf_id]["title_cn"],
                    }
                    for leaf_id in leaf_specs["leaf_ids"]
                ]
                level1_node["leaves"] = leaves
            else:
                for child in spec["children"]:
                    batch_title = self.cache["batch_labels"][child["fingerprint"]]["title_en"]
                    batch_node = {
                        "id": f"batch:{child['node_id']}",
                        **self._node_metadata(child["node_id"], nodes),
                        "title_en": batch_title,
                        "title_cn": translations[batch_title],
                        "parent_id": spec["node_id"],
                        "leaves": [
                            {
                                "id": f"leaf:{leaf_records[leaf_id]['uid']}",
                                "node_id": leaf_id,
                                "playlist_id": leaf_records[leaf_id]["playlist_id"],
                                "uid": leaf_records[leaf_id]["uid"],
                                "title_en": leaf_records[leaf_id]["title_en"],
                                "title_cn": leaf_records[leaf_id]["title_cn"],
                            }
                            for leaf_id in child["leaf_ids"]
                        ],
                    }
                    level1_node["children"].append(batch_node)

            categories[category_index - 1]["count"] += spec["count"]
            categories[category_index - 1]["children"].append(level1_node)

        # Coverage and uniqueness validation before exposing output.
        output_uids = []
        for category in categories:
            for parent in category["children"]:
                if parent.get("leaves"):
                    output_uids.extend(leaf["uid"] for leaf in parent["leaves"])
                for child in parent.get("children", []):
                    output_uids.extend(leaf["uid"] for leaf in child.get("leaves", []))
        expected_uids = [record["uid"] for record in leaf_records.values()]
        if len(output_uids) != len(expected_uids):
            raise RuntimeError(
                f"Taxonomy coverage mismatch: output={len(output_uids)}, expected={len(expected_uids)}"
            )
        if len(set(output_uids)) != len(output_uids):
            raise RuntimeError("Duplicate UID detected in taxonomy output")
        if set(output_uids) != set(expected_uids):
            raise RuntimeError("Taxonomy output UID set does not match source leaves")
        if any(not category["children"] for category in categories):
            raise RuntimeError("Empty top-level category detected")

        batch_sizes = [spec["count"] for spec in batch_specs]
        level1_sizes = [spec["count"] for spec in level1_specs]
        return {
            "version": "taxonomy-v2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "parent_max_size": self.parent_max_size,
                "batch_max_size": self.batch_max_size,
                "top_category_count_requested": self.top_category_count,
                "naming_batch_size": self.naming_batch_size,
                "strategy": "adaptive_recursive_cut",
            },
            "stats": {
                "leaf_count": len(leaf_records),
                "top_category_count": len(categories),
                "level_1_count": len(level1_specs),
                "batch_count": len(batch_specs),
                "level_1_size": self._size_stats(level1_sizes),
                "batch_size": self._size_stats(batch_sizes),
                "uid_coverage": len(set(output_uids)),
            },
            "root": {
                "node_id": root_id,
                "count": len(leaf_records),
                "children": [category["id"] for category in categories],
            },
            "categories": categories,
            "node_labels": {
                "level_1": {
                    spec["node_id"]: {
                        "title_en": spec["title_en"],
                        "title_cn": translations[spec["title_en"]],
                    }
                    for spec in level1_specs
                },
                "batches": {
                    spec["node_id"]: {
                        "title_en": self.cache["batch_labels"][spec["fingerprint"]]["title_en"],
                        "title_cn": translations[
                            self.cache["batch_labels"][spec["fingerprint"]]["title_en"]
                        ],
                    }
                    for spec in batch_specs
                },
            },
        }

    @staticmethod
    def _size_stats(values: list[int]) -> dict[str, float | int]:
        if not values:
            return {"min": 0, "max": 0, "avg": 0}
        return {
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / len(values), 2),
        }
