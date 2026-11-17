from personal_store import PersonalStore
from DrissionPage import Chromium, ChromiumOptions
from DrissionPage.common import Actions
import time
import argparse
from traceback import format_exc
from deep_translator import GoogleTranslator

def main(args):
    proxies = {
        "https": '127.0.0.1:1080',
        "http": '127.0.0.1:1080'
    }
    translator = GoogleTranslator(source="en", target="zh-CN", proxies=proxies)
    ps = PersonalStore("https://hc.lxc.nya.je:24687/", args.storetoken)
    # ps.save_json("xxx.json", data, indent=None)
    version = "v1"
    filename = "innertune_list.json"
    translate_dict_filename = "innertune_en_cn.json"
    is_find_meta = True
    data = ps.try_load_json(filename)
    if data:
        if data["version"] == version:
            if time.time() - float(data["time"]) < 7 * 24 * 3600:
                is_find_meta = False
    translate_dict = ps.try_load_json(translate_dict_filename)
    if not translate_dict:
        translate_dict = {}
    
    co = ChromiumOptions()
    co.set_argument('--no-sandbox')
    co.set_argument('--disable-gpu')
    co.set_argument('--window-size', '1920,1080')
    co.set_argument('--start-maximized')
    browser = Chromium(addr_or_opts=co)
    tab = browser.latest_tab
    ac = Actions(tab)
    tab.set.window.full()
    
    url = "https://innertune.com/affirmations"
    tab.get(url)
    
    try:
        time.sleep(5)
        if is_find_meta:
            print("开始查询Playlist")
            lists_a = tab.eles("css:div.grid > a.text-left", timeout = 1.5)
            print(f"找到了{len(lists_a)}个格子")
            lists = []
            for block in lists_a:
                url_a = block.attr("href")
                is_coming_soon = False
                try:
                    span = block.ele("css:div.relative > span", timeout = 0.1)
                    if span.text == "Coming Soon":
                        is_coming_soon = True
                except:
                    pass
                title_en = block.ele("css:div.p-4 > h3", timeout = 0.1).text
                intro_en = block.ele("css:div.p-4 > p", timeout = 0.1).text
                list_dict = {
                    "url": url_a,
                    "is_coming_soon": is_coming_soon,
                    "title_en": title_en,
                    "intro_en": intro_en
                }
                lists.append(list_dict)
            print(f"完成了{len(lists)}个格子的信息收集")
            print(lists[:3])
            for i, l in enumerate(lists):
                l["title_cn"], l["intro_cn"] = translator.translate_batch([l["title_en"], l["intro_en"]])
                time.sleep(0.25)
                lists[i] = l
            print(f"完成了{len(lists)}个格子的信息翻译")
            print(lists[:3])
            data = {
                "url": "https://innertune.com/affirmations",
                "version": version,
                "time": time.time(),
                "list": lists
            }
            ps.save_json(filename, data, indent=None)
    except:
        error_msg = format_exc()
        print(error_msg)
    print("执行完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DrissionPage Action")
    parser.add_argument("--pterclubcookies", required=True, help="Auth Token")
    parser.add_argument("--tgbottoken", required=True, help="TG BOT Token")
    parser.add_argument("--tgchatid", required=True, help="TG Chat ID")
    parser.add_argument("--fnyuser", required=True, help="FNY User")
    parser.add_argument("--fnypswd", required=True, help="FNY Password")
    parser.add_argument("--proxy", required=False, help="Proxy URL")
    parser.add_argument("--resendkey", required=True, help="Resend Key")
    parser.add_argument("--storetoken", required=True, help="Store Token")

    args = parser.parse_args()
    
    main(args)