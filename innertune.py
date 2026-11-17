from personal_store import PersonalStore
from DrissionPage import Chromium, ChromiumOptions
from DrissionPage.common import Actions
import time
import argparse
from traceback import format_exc
from translator import Translator
import requests
import io

def capture_and_send(page, title, time_wait=1.5):
    time.sleep(time_wait)
    BOT_TOKEN = args.tgbottoken
    CHAT_ID = args.tgchatid
    
    print("正在截取可视窗口图片（返回字节流）...")
    img_bytes = page.get_screenshot(as_bytes='webp', full_page=False)
    
    print("正在通过 Telegram 发送图片...")
    telegram_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    
    # 文本参数
    data = {
        "chat_id": CHAT_ID,
        "caption": f"DrissionPage截取页面 - {title}"
    }
    
    # 将内存中的字节数据转换为 Telegram requests 接收的文件格式
    files = {
        "photo": ("screenshot.webp", io.BytesIO(img_bytes), "image/webp")
    }
    
    # 发送请求
    response = requests.post(telegram_url, data=data, files=files)
    
    result = response.json()
    if result.get("ok"):
        print("✅ 图片发送成功！")
    else:
        print(f"❌ 发送失败，原因: {result.get('description')}")

def main(args):
    
    translator = Translator(quantization="Q4_K_M", threads=4)
    translator.start()
    
    ps = PersonalStore("https://hc.lxc.nya.je:24687/", args.storetoken)
    version = "v2"
    filename = "innertune_list.json"
    translate_dict_filename = "innertune_en_cn.json"
    is_find_meta = True
    data = ps.try_load_json(filename)
    if data:
        if data["version"] == version:
            if time.time() - float(data["time"]) < 7 * 24 * 3600:
                is_find_meta = False
                count = len(data["list"])
    else:
        data = {}
        count = 0
    
    re_translate = False
    translate_dict = ps.try_load_json(translate_dict_filename)
    if not translate_dict:
        translate_dict = {}
    if re_translate:
        translate_dict = {}
        is_find_meta = True
    
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
    time.sleep(5)
    
    # print("点击接收Cookies")
    # ac.move_to((432, 136), duration=0.5).move_to((1428, 988), duration=0.5).click()
    
    try:
        if is_find_meta:
            print("开始查询Playlist")
            lists_a = tab.eles("css:div.grid > a.text-left", timeout = 1.5)
            print(f"找到了{len(lists_a)}个格子")
            lists = []
            for block in lists_a:
                url_a = block.attr("href")
                is_coming_soon = False
                try:
                    span = block.ele("css:div.relative > span.tracking-wider", timeout = 0.001)
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
                    l["title_cn"] = translator.translate(l["title_en"])
                    translate_dict[l["title_en"]] = l["title_cn"]
                else:
                    l["title_cn"] = translate_dict[l["title_en"]]
                if l["intro_en"]:
                    if (l["intro_en"] not in translate_dict):
                        l["intro_cn"] = translator.translate(l["intro_en"])
                        translate_dict[l["intro_en"]] = l["intro_cn"]
                    else:
                        l["intro_cn"] = translate_dict[l["intro_en"]]
                else:
                    l["intro_cn"] = ""
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
            count = len(data["list"])
            ps.save_json(filename, data, indent=None)
            ps.save_json(translate_dict_filename, translate_dict, indent=None)
        # add content
        for i, page_url in enumerate(data["list"]):
            item = data["list"][page_url]
            print(f"进入: {i+1}/{count} {page_url}")
            # 跳过Coming Soon
            if item["is_coming_soon"]:
                print(f"跳过: {i+1}/{count} Coming Soon")
                continue
            # 跳过已完成
            if "contents" in data:
                if page_url in data["contents"]:
                    print(f"跳过: {i+1}/{count} 已完成")
                    continue
                
            tab.get(page_url)
            time.sleep(5)
            try:
                btn = tab.ele("css:div.mt-6 > button", timeout=1.5)
                btn.click(by_js=True, timeout=1.5)
            except:
                try:
                    cs = tab.ele("css:div.flex-1 > div.mb-6 > span.text-amber-400")
                    if cs.text == "Coming Soon":
                        data["list"][page_url]["is_coming_soon"] = True
                        print(f"发现: {i+1}/{count} Coming Soon")
                        ps.save_json(filename, data, indent=None)
                        continue
                except:
                    print(f"警告: {i+1}/{count} 未点击按钮")
                    pass
            stns = tab.eles("css:div.relative > div.space-y-3 > div.items-start > p.text-sm", timeout=1.5)
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
                print(f"完成: {i+1}/{count} 发现 {len(sentences)} 条肯定句")
            else:
                print(f"警告: {i+1}/{count} {page_url} 没有点击按钮？仅发现 {len(sentences)} 条肯定句")
    except:
        error_msg = format_exc()
        print(error_msg)
    print("执行完毕！")
    translator.close()


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