from DrissionPage import ChromiumPage, ChromiumOptions
import os
import random
from traceback import format_exc

cookies_str = os.getenv("MY_WEB_COOKIES", "session_id=default_value")
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
  feed_str += "兑换一次3,200魔力值成功！\n"
  page.wait(3)
except:
  error_msg = format_exc()
  feed_str += "点击兑换失败:\n"
  feed_str += error_msg
  feed_str += "\n\n"

page.get_screenshot(name='pic.png', as_bytes="png", full_page=True)



page.quit()
