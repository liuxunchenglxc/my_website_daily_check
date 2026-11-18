LLAMA_REPO = "ggml-org/llama.cpp"
LLAMA_VERSION = "b12503"
LLAMA_FILE = f"llama-{LLAMA_VERSION}-bin-ubuntu-x64.tar.gz"

import requests

def update_llama_version():
    global LLAMA_VERSION, LLAMA_FILE
    try:
        LLAMA_VERSION = requests.get(
            "https://api.github.com/repos/yt-dlp/yt-dlp/releases"
        ).json()[0]["tag_name"]
    except:
        LLAMA_VERSION = "b12503"
    LLAMA_FILE = f"llama-{LLAMA_VERSION}-bin-ubuntu-x64.tar.gz"
    print("LLAMA_VERSION is", LLAMA_VERSION)
    print("LLAMA_FILE is", LLAMA_FILE)