"""從 screenshots/app_log.json 產生 RAG02 notebook §11（Gradio 對話結果截圖）的 markdown。"""
import json, re
RAW = "https://raw.githubusercontent.com/arvin611ch/taica-lab07-sunzi-rag/main/screenshots/"
log = {e["event"]: e for e in json.load(open("screenshots/app_log.json", encoding="utf-8"))}


def first_quote_line(ans):
    m = re.search(r"【兵法曰】\s*(.+)", ans)
    return (m.group(1).strip() if m else ans.strip().splitlines()[0])[:120]


def qa(e):
    return (f"> **主公：** {e['q']}  \n> **軍師：** {first_quote_line(e['a'])}  \n> **核對：** {e['status']}")


e1, e2, e3, e4 = log["ask_1"], log["follow_up"], log["rewrite_on"], log["off_topic"]
used3 = e3["used"].replace("實際拿去檢索的句子：", "").strip()

md = f"""
## 10. Gradio 對話結果（實際操作截圖）

截圖是在本機跑 `app.py`（和這本 notebook 同一份程式，模型 `ollama:qwen2.5:14b`）、用 Playwright 真的打字送出、按按鈕後擷取的；每一張對應的完整對話與核對結果在 repo 的 `screenshots/app_log.json`。流程：開局 → 問一題 → 追問 → 重新開局並打開「翻成兵法語再檢索」→ 再重新開局問一個兵法沒談到的問題。

**(1) 開局畫面。** 聊天區是軍師的開場白；下方是問題框、top-k 滑桿、改寫檢索開關、重新開局；再下面是「核對結果」一行與可收合的「軍師翻閱的原文」。

![start]({RAW}app_1_start.png)

**(2) 問「期中考五科只夠讀三科怎麼取捨」。** 回答固定四段【兵法曰】【軍師解】【方略】【一言】；核對那一行顯示每句引文是否真的是原文、編號是否在這輪檢索到的段落裡；下方表格與原文區就是這一輪送給模型的三段。

{qa(e1)}

![answer]({RAW}app_2_answer.png)

**(3) 追問「方略第二條具體怎麼做」。** 這一輪的檢索用的是追問句本身（所以找到的段落不一定相關），但軍師有對話歷史（`gr.State`），接著上一輪的方略往下講。

{qa(e2)}

![follow-up]({RAW}app_3_followup.png)

**(4) 重新開局，打開「先把問題翻成兵法語再檢索」。** 模型先把「跟室友吵架要不要正面對決」翻成兵法風格的句子（原文區第一行會顯示實際拿去檢索的句子），再用那句去找段落：

> 實際拿去檢索的句子：{used3}

{qa(e3)}

![rewrite]({RAW}app_4_rewrite.png)

**(5) 問「今天晚餐吃什麼」。** FAISS 照樣回傳三段最像的，但軍師應該回「兵法未載此事」而不是硬套：

{qa(e4)}

![off topic]({RAW}app_5_offtopic.png)
"""
open("tools/sections/shots.md", "w", encoding="utf-8").write(md.strip("\n") + "\n")
print("shots.md written")
