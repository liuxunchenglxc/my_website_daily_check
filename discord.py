from calendar import c

from DrissionPage import Chromium, ChromiumOptions
from DrissionPage.common import Actions
import resend
import pyperclip
import os
import random
from traceback import format_exc
import io
import requests
import argparse
import time
from datetime import datetime, timedelta
import hashlib
import zipfile
import json

parser = argparse.ArgumentParser(description="DrissionPage Action")
parser.add_argument("--tgbottoken", required=True, help="TG BOT Token")
parser.add_argument("--tgchatid", required=True, help="TG Chat ID")
parser.add_argument("--discorduser", required=True, help="Discord User")
parser.add_argument("--discordpswd", required=True, help="Discord Password")
parser.add_argument("--proxy", required=False, help="Proxy URL")
parser.add_argument("--resendkey", required=True, help="Resend Key")

args = parser.parse_args()

def load_netscape_cookies(cook: str):
    cookies_list = []
    min_expiry = None
    for line in cook.splitlines():
        # 忽略注释行和空行
        if line.startswith('#') or not line.strip():
            continue
        
        # Netscape 格式以制表符 \t 分隔
        parts = line.strip().split('\t')
        if len(parts) < 7:
            continue
            
        # 解析 Netscape 字段
        cookie_dict = {
            'domain': parts[0],
            'flag': parts[1],    # 通常是 TRUE/FALSE，DrissionPage 传 dict 时可省略
            'path': parts[2],
            'secure': parts[3] == 'TRUE',
            'expiry': int(parts[4]) if parts[4].isdigit() else None,
            'name': parts[5],
            'value': parts[6]
        }
        
        # print(f"解析到 Cookie: {cookie_dict['name']}，过期时间: {cookie_dict['expiry']}")
        
        if min_expiry is None:
            min_expiry = cookie_dict['expiry']
        elif cookie_dict['expiry'] > 0 and cookie_dict['expiry'] < min_expiry:
            min_expiry = cookie_dict['expiry']
            
        cookies_list.append(cookie_dict)
            
    return cookies_list, min_expiry

def send_tg_msg(text):
    BOT_TOKEN = args.tgbottoken
    CHAT_ID = args.tgchatid
    telegram_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    
    # 文本参数
    data = {
        "chat_id": CHAT_ID,
        "text": text
    }
    
    # 发送请求
    response = requests.post(telegram_url, data=data)
    
    result = response.json()
    if result.get("ok"):
        print("✅ 消息发送成功！")
    else:
        print(f"❌ 发送失败，原因: {result.get('description')}")

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

def send_resend_email(sub_url, status: bool):
    prefix = random.choice(["呜喵！", "主人主人！", "喵呜~", "报告主人！"])
    suffix = random.choice(["要好好使用喵~", "这是人家的努力成果喵！", "喵喵喵~"])

    if status:
        subject_text = f"主人的飞鸟云订阅更新成功了喵！"
        body_content = (
            f"<p>{prefix}</p>\n\n"
            f"<p>人家已经帮主人把飞鸟云的订阅链接取回来啦喵！</p>\n"
            f"<p>链接就在下面，请主人查收喵：</p>\n\n"
            f"<p>{sub_url}</p>\n\n"
            f"<p>{suffix}</p>"
        )
    else:
        subject_text = f"呜呜...飞鸟云订阅更新出错了喵..."
        body_content = (
            f"<p>对不起主人喵...人家没能拿到链接...</p>\n"
            f"<p>报错信息是这个喵：</p>\n"
            f"<p>{sub_url}</p>\n"
            f"<p>主人快去检查一下设置喵！</p>"
        )
    
    params: resend.Emails.SendParams = {
    "from": "Github Action 猫娘 <onboarding@resend.dev>",
    "to": ["liuxunchenglxc@163.com"],
    "subject": subject_text,
    "html": body_content
    }

    email = resend.Emails.send(params)
    print("邮件发送结果：")
    print(email)

def is_repeat_url(sub_url, status):
    if status:
        string_hash = hashlib.sha256(sub_url.encode()).hexdigest()
        if os.path.exists("fnyhash.txt"):
            with open("fnyhash.txt", "r", encoding="utf-8") as f:
                old_hash = f.read()
            if old_hash == string_hash:
                return True
        with open("fnyhash.txt", "w", encoding="utf-8") as f:
            f.write(string_hash)
    return False

def create_proxy_auth_extension(proxy_host, proxy_port, username, password):
    manifest_json = {
        "version": "1.0.0",
        "manifest_version": 2,
        "name": "Proxy Auth",
        "permissions": ["proxy", "tabs", "unlimitedStorage", "storage"],
        "background": {"scripts": ["background.js"]},
        "minimum_chrome_version": "22.0.0"
    }
 
    auth_script = f"""
var config = {{
    mode: "fixed_servers",
    rules: {{
        singleProxy: {{
            scheme: "http",
            host: "{proxy_host}",
            port: {int(proxy_port)}
        }},
        bypassList: ["localhost"]
    }}
}};
chrome.proxy.settings.set({{value: config, scope: "regular"}}, function() {{}});
 
chrome.webRequest.onAuthRequired.addListener(
    (details) => {{
        return {{
            authCredentials: {{ username: "{username}", password: "{password}" }}
        }};
    }},
    {{urls: ["<all_urls>"]}},
    ['blocking']
);
"""
 
    # 构建文件夹
    os.makedirs('proxy_auth_plugin', exist_ok=True)
    with open('proxy_auth_plugin/manifest.json', 'w') as f:
        json.dump(manifest_json, f)
    with open('proxy_auth_plugin/background.js', 'w') as f:
        f.write(auth_script)
    return 'proxy_auth_plugin'

# cookies_str, min_expiry = load_netscape_cookies(args.pterclubcookies)
co = ChromiumOptions()
proxy: str = args.proxy
uspw, hspt = proxy.split("://")[1].strip('/').split('@')
user, pswd = uspw.split(':')
host, port = hspt.split(':')
plugin_path = create_proxy_auth_extension(host, port, user, pswd)
co.add_extension(plugin_path)
# co.add_extension("buster")
co.set_argument('--no-sandbox')
co.set_argument('--disable-gpu')
co.set_argument('--window-size', '1920,1080')
co.set_argument('--start-maximized')
browser = Chromium(addr_or_opts=co)

resend.api_key = args.resendkey

tab = browser.latest_tab
ac = Actions(tab)
# page.set.cookies(cookies_str)
tab.set.window.full()

url = "https://chromewebstore.google.com/detail/hcaptcha-solver-by-noneca/kahepbmokpbbkfognbmimpmnlfibcolf"
tab.get(url)

try:
    time.sleep(5)
    ac.move_to((432, 136), duration=0.5).move_to((1261,156), duration=0.5).click()
    capture_and_send(tab, f"Chrome Web Store 页面", time_wait=1.5)
except:
    error_msg = format_exc()
    print(error_msg)

# url = "https://discord.com/login"
# tab.get(url)

# try:
#     time.sleep(5)
#     capture_and_send(tab, f"https://discord.com/login", time_wait=1.5)
#     user = tab.ele('css:input[type="text"]', timeout=1.5)
#     pswd = tab.ele('css:input[type="password"]', timeout=1.5)
#     login = tab.ele('css:button[type="submit"]', timeout=1.5)
#     if user and pswd and login:
#         user.input(args.discorduser, clear=True, by_js=False)
#         pswd.input(args.discordpswd, clear=True, by_js=False)
#         login.click(by_js=True, timeout=1.5)
#         print("点击登录按钮")
#         ac.move_to((432, 136), duration=0.5).move_to((840,653), duration=0.5).click()
#         time.sleep(5)
#         capture_and_send(tab, f"Discord 登录操作完成", time_wait=1.5)
# except:
#   error_msg = format_exc()
#   print(error_msg)
# #   send_resend_email(error_msg, status=False)
# #   capture_and_send(tab, f"飞鸟云订阅更新错误，请查看邮件", time_wait=1.5)

print("执行完毕！")

# try:
#     time.sleep(5)
#     print("点击CF框")
#     ac.move_to((432, 136), duration=0.5).move_to((532, 336), duration=0.5).click()
#     user = tab.ele('css:div.form-group > input[type="text"]', timeout=5)
#     pswd = tab.ele('css:div.form-group > input[type="password"]', timeout=1.5)
#     login = tab.ele('css:button[type="submit"]', timeout=1.5)
#     user.input(args.discorduser, clear=True, by_js=True)
#     pswd.input(args.discordpswd, clear=True, by_js=True)
#     login.click(by_js=True, timeout=1.5)
#     print("点击登录按钮")
#     try:
#         time.sleep(2)
#         close = tab.ele('css:button.ant-modal-close > span > i', timeout=1.5)
#         close.click(by_js=True, timeout=1.5)
#         print("关闭公告")
#     except:
#         pass
#     sub_1 = tab.ele('text=一键订阅', timeout=1.5)
#     sub_1.click(by_js=True, timeout=1.5)
#     sub_2 = tab.ele('text=复制订阅地址', timeout=1.5)
#     sub_2.click(by_js=True, timeout=1.5)
#     sub_url = pyperclip.paste()
#     if not is_repeat_url(sub_url, True):
#         print("[+] 检测到为全新URL。")
#         capture_and_send(tab, f"飞鸟云全新订阅地址：{sub_url}", time_wait=1.5)
#         send_resend_email(sub_url, status=True)
#     else:
#         print("[=] 检测到为重复URL。")
#     tab.wait(3)
# except:
#   error_msg = format_exc()
#   print(error_msg)
#   send_resend_email(error_msg, status=False)
#   capture_and_send(tab, f"飞鸟云订阅更新错误，请查看邮件", time_wait=1.5)
# print("执行完毕！")