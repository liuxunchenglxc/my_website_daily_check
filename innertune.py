from personal_store import PersonalStore
from DrissionPage import Chromium, ChromiumOptions
from DrissionPage.common import Actions
import time
import argparse
from traceback import format_exc
import argostranslate.package
import argostranslate.translate

def translate(text):
    return argostranslate.translate.translate(
        text, "en", "zh"
    )

def main(args):
    # proxies = {
    #     "https": '127.0.0.1:1080',
    #     "http": '127.0.0.1:1080'
    # }

    argostranslate.package.update_package_index()
    # https://www.argosopentech.com/argospm/index/
    packages = argostranslate.package.get_available_packages()
    package = next(
        p for p in packages
        if p.from_code == "en" and p.to_code == "zh"
    )
    argostranslate.package.install_from_path(package.download())
    
    ps = PersonalStore("https://hc.lxc.nya.je:24687/", args.storetoken)
    # ps.save_json("xxx.json", data, indent=None)
    version = "v2"
    filename = "innertune_list.json"
    translate_dict_filename = "innertune_en_cn.json"
    is_find_meta = True
    data = ps.try_load_json(filename)
    if data:
        if data["version"] == version:
            if time.time() - float(data["time"]) < 7 * 24 * 3600:
                is_find_meta = False
    else:
        data = {}
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
                    span = block.ele("css:div.relative > span", timeout = 0.001)
                    if span.text == "Coming Soon":
                        is_coming_soon = True
                except:
                    pass
                title_en = block.ele("css:div.p-4 > h3", timeout = 0.001).text
                intro_en = block.ele("css:div.p-4 > p", timeout = 0.001).text
                list_dict = {
                    "url": url_a,
                    "is_coming_soon": is_coming_soon,
                    "title_en": title_en,
                    "intro_en": intro_en
                }
                if title_en:
                    lists.append(list_dict)
            print(f"完成了{len(lists)}个格子的信息收集")
            print(lists[:3])
            for i, l in enumerate(lists):
                if not l["title_en"]:
                    continue
                if (l["title_en"] not in translate_dict):
                    l["title_cn"] = translate(l["title_en"])
                    translate_dict[l["title_en"]] = l["title_cn"]
                else:
                    l["title_cn"] = translate_dict[l["title_en"]]
                if l["intro_en"]:
                    if (l["intro_en"] not in translate_dict):
                        l["intro_cn"] = translate(l["intro_en"])
                        translate_dict[l["intro_en"]] = l["intro_cn"]
                    else:
                        l["intro_cn"] = translate_dict[l["intro_en"]]
                else:
                    l["intro_cn"] = ""
                time.sleep(1)
                lists[i] = l
                print(f"已翻译: {i+1}/{len(lists)}")
            print(f"完成了{len(lists)}个格子的信息翻译")
            print(lists[:3])
            data["url"] = "https://innertune.com/affirmations"
            data["version"] = version
            data["time"] = time.time()
            list_dict = {}
            for l in lists:
                list_dict[l["url"]] = l
            data["list"] = list_dict
            ps.save_json(filename, data, indent=None)
            ps.save_json(translate_dict_filename, translate_dict, indent=None)
        # add content
        for i, item in enumerate(data["list"]):
            if item["is_coming_soon"]:
                print(f"跳过Coming Soon: {i+1}/{len(lists)}")
                continue
            page_url = item["url"]
            # 跳过已完成
            if "contents" in data:
                if page_url in data:
                    print(f"跳过已完成: {i+1}/{len(lists)}")
                    continue
            tab.get(page_url)
            time.sleep(5)
            btn = tab.ele("css:div.mt-6 > button", timeout=1.5)
            btn.click(by_js=True, timeout=1.5)
            stns = tab.eles("css:div.flex > p", timeout=1.5)
            sentences = []
            for s in stns:
                t = s.text
                if t:
                    sentences.append(t)
            if len(sentences) > 8:
                d = {
                    "url": page_url,
                    "sentences": sentences,
                    "time": time.time(),
                }
                if "contents" not in data:
                    data["contents"] = {}
                data["contents"][page_url] = d
                ps.save_json(filename, data, indent=None)
                print(f"完成: {i+1}/{len(lists)}")
            else:
                print(f"警告: {i+1}/{len(lists)} {page_url} 没有点击按钮？仅发现 {len(sentences)} 条肯定句")
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