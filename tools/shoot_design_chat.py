"""用 Playwright 真的在聊天室裡送出問題、等 AI 回答、每輪截圖，並把對話存成 markdown。"""
import sys, time, json, os
from playwright.sync_api import sync_playwright

URL = sys.argv[1]
OUT = sys.argv[2]
PROMPTS = json.load(open(sys.argv[3], encoding="utf-8"))
os.makedirs(OUT, exist_ok=True)
log = []

def bot_texts(page):
    els = page.locator('[data-testid="bot"]')
    return [els.nth(i).inner_text().strip() for i in range(els.count())]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1280, "height": 1120}, device_scale_factor=1.5)
    page.goto(URL)
    page.wait_for_selector('textarea', timeout=60000)
    for i, prompt in enumerate(PROMPTS, 1):
        before = len(bot_texts(page))
        box = page.locator('textarea').first
        box.fill(prompt)
        box.press("Enter")
        # 等到新的 AI 訊息出現、而且內容 2 秒內沒再變化
        last, stable, t0 = None, 0, time.time()
        while time.time() - t0 < 900:
            time.sleep(1.0)
            texts = bot_texts(page)
            cur = texts[-1] if len(texts) > before and texts[-1] else None
            if cur and cur == last:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            last = cur
        reply = last or "(no reply)"
        log.append({"round": i, "prompt": prompt, "reply": reply})
        # 把畫面捲到這輪的提問處再截圖
        user_els = page.locator('[data-testid="user"]')
        user_els.nth(user_els.count() - 1).scroll_into_view_if_needed()
        time.sleep(0.5)
        page.screenshot(path=f"{OUT}/design_chat_{i}.png")
        print(f"round {i} done, {len(reply)} chars", flush=True)
    browser.close()

with open(f"{OUT}/design_chat_log.json", "w", encoding="utf-8") as f:
    json.dump(log, f, ensure_ascii=False, indent=2)
