from DrissionPage import Chromium, ChromiumOptions
from DrissionPage.common import Actions
import os
import random
from traceback import format_exc
import io
import requests
import argparse
import time
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description="DrissionPage Action")
parser.add_argument("--pterclubcookies", required=True, help="Auth Token")
parser.add_argument("--tgbottoken", required=True, help="TG BOT Token")
parser.add_argument("--tgchatid", required=True, help="TG Chat ID")
parser.add_argument("--fnyuser", required=True, help="FNY User")
parser.add_argument("--fnypswd", required=True, help="FNY Password")
parser.add_argument("--proxy", required=False, help="Proxy URL")


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
    img_bytes = page.get_screenshot(as_bytes=True, full_page=False)
    
    print("正在通过 Telegram 发送图片...")
    telegram_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    
    # 文本参数
    data = {
        "chat_id": CHAT_ID,
        "caption": f"DrissionPage截取页面 - {title}"
    }
    
    # 将内存中的字节数据转换为 Telegram requests 接收的文件格式
    # 'screenshot.jpg' 是虚拟文件名，告诉 Telegram 这是 JPG 格式
    files = {
        "photo": ("screenshot.jpg", io.BytesIO(img_bytes), "image/jpeg")
    }
    
    # 发送请求
    response = requests.post(telegram_url, data=data, files=files)
    
    result = response.json()
    if result.get("ok"):
        print("✅ 图片发送成功！")
    else:
        print(f"❌ 发送失败，原因: {result.get('description')}")

# cookies_str, min_expiry = load_netscape_cookies(args.pterclubcookies)
co = ChromiumOptions()
co.set_argument('--no-sandbox')
co.set_argument('--disable-gpu')
co.set_argument('--window-size', '1920,1080')
co.set_argument('--start-maximized')
browser = Chromium(addr_or_opts=co)

tab = browser.latest_tab
ac = Actions(tab)
# page.set.cookies(cookies_str)
tab.set.window.full()

url = "https://feiniaoyun.xyz/#/login"
tab.get(url)

try:
  time.sleep(5)
  ac.move_to((432, 136), duration=0.5).move_to((532, 336), duration=0.5).click()
  user = tab.ele('css:div.form-group > input[type="text"]', timeout=5)
  pswd = tab.ele('css:div.form-group > input[type="password"]', timeout=1.5)
  login = tab.ele('css:button[type="submit"]', timeout=1.5)
  try:
    user.input(args.fnyuser, clear=True, by_js=True)
    pswd.input(args.fnypswd, clear=True, by_js=True)
    login.click(by_js=True, timeout=1.5)
  except:
    pass
  capture_and_send(tab, "飞鸟云登陆页面", time_wait=1.5)
  tab.wait(3)
except:
  error_msg = format_exc()

capture_and_send(tab, "飞鸟云登陆页面后", time_wait=1.5)

# try:
#   button = page.ele('@@tag()=td@@text()=3,200').next().child()
#   button.click(by_js=True, timeout=1.5)
# #   capture_and_send(page, "兑换点击后", time_wait=1.5)
#   text = "兑换一次3,200魔力值成功！\n"
# #   send_tg_msg(text)
#   print(text)
#   page.wait(3)
# except:
#   error_msg = format_exc()
#   text = "点击兑换失败:\n"
#   text += error_msg
# #   send_tg_msg(text)
#   print(text)

# expiry_time_str = (datetime.fromtimestamp(min_expiry) + timedelta(hours=8)).strftime('%Y年%m月%d日%H时') if min_expiry else "未知"
# capture_and_send(page, f"DrissionPage Actor 执行 pterclub.net 签到和兑换魔力值任务最终结果（过期时间：{expiry_time_str}）", time_wait=1.5)

# send_tg_msg("*DrissionPage Actor 执行完毕！*")