from DrissionPage import ChromiumPage, ChromiumOptions
import os
import random
from traceback import format_exc
import io
import requests
import argparse

parser = argparse.ArgumentParser(description="DrissionPage Action")
parser.add_argument("--pterclubcookies", required=True, help="Auth Token")
parser.add_argument("--tgbottoken", required=True, help="TG BOT Token")
parser.add_argument("--tgchatid", required=True, help="TG Chat ID")

args = parser.parse_args()

def load_netscape_cookies(cook: str):
    cookies_list = []
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
        
        # 过滤掉没有过期时间的无效数据（可选）
        if cookie_dict['expiry'] is None:
            cookie_dict.pop('expiry')
            
        cookies_list.append(cookie_dict)
            
    return cookies_list


def capture_and_send(page):
    # --- 1. 配置信息 ---
    BOT_TOKEN = args.tgbottoken
    CHAT_ID = args.tgchatid
    
    print("正在截取可视窗口图片（返回字节流）...")
    img_bytes = page.get_screenshot(as_bytes=True, full_page=False)
    
    # --- 3. 将字节流转换为文件对象并发送给 Telegram ---
    print("正在通过 Telegram 发送图片...")
    telegram_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    
    # 文本参数
    data = {
        "chat_id": CHAT_ID,
        "caption": f"DrissionPage截取页面"
    }
    
    # 将内存中的字节数据转换为 Telegram requests 接收的文件格式
    # 'screenshot.jpg' 是虚拟文件名，告诉 Telegram 这是 JPG 格式
    files = {
        "photo": ("screenshot.jpg", io.BytesIO(img_bytes), "image/jpeg")
    }
    
    # 发送请求
    response = requests.post(telegram_url, data=data, files=files)
    
    # --- 4. 打印结果 ---
    result = response.json()
    if result.get("ok"):
        print("✅ 图片发送成功！")
    else:
        print(f"❌ 发送失败，原因: {result.get('description')}")

cookies_str = load_netscape_cookies(args.pterclubcookies)
co = ChromiumOptions()
co.set_argument('--no-sandbox')
co.set_argument('--disable-gpu')
page = ChromiumPage(co)
page.set.cookies(cookies_str)
url = "https://pterclub.net/mybonus.php"
page.get(url)

feed_str = ""

try:
  t = random.randint(10, 15)
  button = page.ele('#do-attendance', timeout=t)
  button.click()
  capture_and_send(page)
  feed_str += "点击签到成功！\n"
  page.wait(3)
except:
  error_msg = format_exc()
  feed_str += "点击签到失败:\n"
  feed_str += error_msg
  feed_str += "\n\n"

try:
  button = page.ele('@@tag()=td@@text()=3,200').next().child()
  button.click()
  capture_and_send(page)
  feed_str += "兑换一次3,200魔力值成功！\n"
  page.wait(3)
except:
  error_msg = format_exc()
  feed_str += "点击兑换失败:\n"
  feed_str += error_msg
  feed_str += "\n\n"

capture_and_send(page)

page.quit()
