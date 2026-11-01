"""用 Playwright 真實操作 Gradio 版「孫子 AI 軍師」：問問題、追問、開啟改寫檢索、問兵法未載的問題，每個階段截圖。
用法：PLAYWRIGHT_BROWSERS_PATH=... python tools/shoot_app.py http://127.0.0.1:7863 screenshots
"""
import sys, time, json, os
from playwright.sync_api import sync_playwright

URL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
log = []
PLACEHOLDER = "例如：期中考快到了，五科只夠認真讀三科，該怎麼取捨？"


def bots(page):
    els = page.locator('[data-testid="bot"]')
    return [els.nth(i).inner_text().strip() for i in range(els.count())]


def status_text(page):
    el = page.locator("text=/(🔎|（尚未開示）|（已重新開局）|（請先輸入問題）)/").first
    return el.inner_text().strip() if el.count() else ""


def wait_until(fn, timeout=1200, every=1.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(every)
    raise TimeoutError("timed out")


def ask(page, text, label):
    before_n, before_status = len(bots(page)), status_text(page)
    box = page.get_by_placeholder(PLACEHOLDER)
    box.fill(text)
    box.press("Enter")
    reply = wait_until(lambda: (lambda b: b[-1] if len(b) > before_n and b[-1] else None)(bots(page)))
    # 聊天、核對結果、表格、原文是同一個事件一起更新的；回覆出現後稍等一下再讀核對那一行即可
    time.sleep(2.0)
    status = wait_until(lambda: (lambda s: s if s.startswith("🔎") else None)(status_text(page)), timeout=60)
    passages = page.locator("text=實際拿去檢索的句子").first
    used = passages.inner_text().strip() if passages.count() else ""
    log.append({"event": label, "q": text, "a": reply, "status": status, "used": used})
    print(f"  [{label}] Q: {text}\n  A: {reply[:100].replace(chr(10), ' ')}…\n  status: {status}", flush=True)
    return reply


def shot(page, name):
    time.sleep(0.8)
    page.screenshot(path=f"{OUT}/{name}.png", full_page=True)
    print("shot", name, flush=True)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=1.5)
    page.goto(URL)
    page.wait_for_selector("text=孫子 AI 軍師", timeout=60000)
    shot(page, "app_1_start")

    ask(page, "期中考快到了，五科只夠認真讀三科，該怎麼取捨？", "ask_1")
    shot(page, "app_2_answer")

    ask(page, "方略的第二條具體要怎麼做？可以給我一個今天就能開始的版本嗎？", "follow_up")
    shot(page, "app_3_followup")

    page.get_by_role("button", name="重新開局 🔄").click()
    time.sleep(1.5)
    page.get_by_label("先把問題翻成兵法語再檢索（多一次模型呼叫）").check()
    ask(page, "跟室友吵架了，要不要正面對決？", "rewrite_on")
    shot(page, "app_4_rewrite")

    page.get_by_label("先把問題翻成兵法語再檢索（多一次模型呼叫）").uncheck()
    page.get_by_role("button", name="重新開局 🔄").click()
    time.sleep(1.5)
    ask(page, "今天晚餐吃什麼比較好？", "off_topic")
    shot(page, "app_5_offtopic")
    browser.close()

json.dump(log, open(f"{OUT}/app_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("done")
