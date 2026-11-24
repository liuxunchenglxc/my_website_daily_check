#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 Hy-MT2-7B 优化 InnerTune 翻译记忆。

只读取 innertune_list_analysis.json 和 innertune_en_cn.json，
只写回 innertune_en_cn.json。需要与 translator.py、llama_cpp.py、
personal_store.py 放在同一目录。
"""

import json
import os
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

ANALYSIS_FILE = "innertune_list_analysis.json"
MEMORY_FILE = "innertune_en_cn.json"
MODEL_REPO = "tencent/Hy-MT2-7B-GGUF"
MODEL_FILE = "Hy-MT2-7B-Q4_K_M.gguf"
STORE_BASE_URL = os.getenv("STORE_BASE_URL", "https://bh.lxc.nya.je:25570/")


def clean(value):
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""


def translate_triplet(translator, source):
    prompt = f"""请将同一个 InnerTune playlist 的三项英文内容翻译成简体中文。
三项内容属于同一个主题，必须结合全部上下文理解词义，尤其注意标题中的多义词。
例如 sleep 是“睡眠”，dream 才是“梦/梦想”，具体含义以另外两项内容为准。

要求：
1. 准确、自然、简洁，不要逐词硬译，也不要擅自发挥。
2. TITLE 是主题标题，INTRO 是主题介绍，CORE 是第一人称肯定句。
3. CORE 保留原文人称、时态、语气及肯定/否定关系。
4. 不遗漏原文信息，不添加原文没有的含义。
5. 原文为空时，对应译文也必须为空。
6. 只输出以下三个标签，各占一行；保留标签，不要解释或添加其他内容：
<TITLE>标题译文</TITLE>
<INTRO>简介译文</INTRO>
<CORE>核心句译文</CORE>

待翻译内容：
<TITLE>{source['title']}</TITLE>
<INTRO>{source['intro']}</INTRO>
<CORE>{source['core_sentence']}</CORE>"""

    body = {
        # Hy-MT2 官方推荐不设置默认 system_prompt。
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.6,
        "top_k": 20,
        "repetition_penalty": 1.05,
        "max_tokens": 1024,
        "stream": False,
    }
    request = urllib.request.Request(
        f"{translator.url}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=translator.timeout) as response:
            raw = json.load(response)["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from exc

    raw = re.sub(r"^\s*```(?:xml|text)?\s*|\s*```\s*$", "", str(raw).strip(), flags=re.I)
    matches = re.findall(r"<(TITLE|INTRO|CORE)>\s*(.*?)\s*</\1>", raw, flags=re.I | re.S)
    result = {key.lower(): clean(value) for key, value in matches}
    if set(result) != {"title", "intro", "core"}:
        raise ValueError(f"模型输出格式不正确：{raw!r}")

    result = {
        "title": result["title"],
        "intro": result["intro"],
        "core_sentence": result["core"],
    }
    for key, english in source.items():
        if english and not result[key]:
            raise ValueError(f"{key} 原文非空，但模型返回了空译文")
        if not english:
            result[key] = ""
    return result


def main(args):
    token = os.getenv("STORE_TOKEN", args.storetoken)
    if not token:
        raise RuntimeError("请在 GitHub Actions Secrets 中设置 STORE_TOKEN")

    # 只从 PersonalStore 读取这两个 JSON。
    from personal_store import PersonalStore
    store = PersonalStore(STORE_BASE_URL, token, timeout=60)
    analysis = store.load_json(ANALYSIS_FILE)
    memory = store.load_json(MEMORY_FILE)
    if not isinstance(analysis, dict) or not isinstance(memory, dict):
        raise TypeError("analysis 和 translation memory 的顶层结构都必须是 JSON object")

    tasks = []
    for url, record in analysis.items():
        if not isinstance(url, str) or not isinstance(record, dict):
            continue
        source = {
            "title": clean(record.get("title_en")),
            "intro": clean(record.get("intro_en")),
            "core_sentence": clean(record.get("core_sentence_en")),
        }
        if any(source.values()):
            tasks.append((url, source))

    max_items = int(os.getenv("MAX_ITEMS", "0"))
    if max_items > 0:
        tasks = tasks[:max_items]
    save_every = max(1, int(os.getenv("SAVE_EVERY", "10")))
    threads = int(os.getenv("HYMT_THREADS", "4"))
    context = int(os.getenv("HYMT_CTX", "4096"))

    print(f"analysis 条目：{len(analysis)}")
    print(f"待翻译 playlist：{len(tasks)}（每个 playlist 一次请求，最多三行）")
    print(f"待翻译英文文本：{sum(bool(v) for _, s in tasks for v in s.values())}")
    if os.getenv("DRY_RUN", "0") == "1":
        for url, source in tasks[:3]:
            print(f"\n{url}\nTITLE: {source['title']}\nINTRO: {source['intro']}\nCORE: {source['core_sentence']}")
        return
    if not tasks:
        print("没有需要翻译的条目。")
        return

    # 复用仓库已有 Translator：下载/启动 llama.cpp、下载模型和关闭服务。
    from translator import Translator
    Translator.HF_REPO = MODEL_REPO
    Translator.MODELS = {"Q4_K_M": MODEL_FILE}
    translator = Translator(
        quantization="Q4_K_M",
        cache_dir=os.getenv("HYMT_CACHE", ".hymt7b"),
        threads=threads,
        context=context,
        timeout=600,
    )

    updates = {}
    failures = []
    succeeded = 0
    try:
        print("正在启动 Hy-MT2-7B……")
        translator.start()
        print("模型已就绪，开始翻译。")

        for index, (url, source) in enumerate(tasks, 1):
            result = None
            error = None
            for attempt in range(3):
                try:
                    result = translate_triplet(translator, source)
                    break
                except Exception as exc:
                    error = exc
                    if attempt < 2:
                        print(f"[重试 {attempt + 1}/2] {url}: {exc}")
                        time.sleep(2 ** (attempt + 1))

            if result is None:
                failures.append((url, str(error)))
                print(f"[失败 {index}/{len(tasks)}] {url}；保留旧译文")
                continue

            succeeded += 1
            for field, english in source.items():
                if english:
                    # 同一英文原文重复出现时，保留第一次结合上下文得到的译文。
                    updates.setdefault(english, result[field])

            print(
                f"[{index}/{len(tasks)}] {url}\n"
                f"  TITLE: {result['title']}\n"
                f"  INTRO: {result['intro']}\n"
                f"  CORE:  {result['core_sentence']}"
            )

            if index % save_every == 0 and updates:
                memory.update(updates)
                store.save_json(MEMORY_FILE, memory, indent=None)
                print(f"[保存] 已更新 {len(updates)} 个不同英文文本")

        memory.update(updates)
        if updates:
            store.save_json(MEMORY_FILE, memory, indent=None)

    finally:
        translator.close()

    print("\n处理完成")
    print(f"成功：{succeeded}/{len(tasks)} 个 playlist")
    print(f"更新：{len(updates)} 个不同英文文本")
    print(f"失败：{len(failures)} 个 playlist")
    print(f"只写回 {MEMORY_FILE}；{ANALYSIS_FILE} 仅作为输入，不会被修改。")
    for url, error in failures:
        print(f"失败项：{url}：{error}")
    if failures:
        raise RuntimeError(f"有 {len(failures)} 个 playlist 翻译失败")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DrissionPage Action")
    parser.add_argument("--storetoken", required=True, help="Store Token")

    args = parser.parse_args()
    
    main(args)
