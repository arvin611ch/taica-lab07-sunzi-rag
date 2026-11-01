"""從 screenshots/design_chat_log.json 產生 RAG02 notebook §1（先跟 AI 討論人設）的 markdown：我的分析＋四張截圖＋全文。"""
import json, re, sys
sys.path.insert(0, "tools")
from check_quotes import load_corpus, check_quotes

RAW = "https://raw.githubusercontent.com/arvin611ch/taica-lab07-sunzi-rag/main/screenshots/"
log = {e["round"]: e for e in json.load(open("screenshots/design_chat_log.json", encoding="utf-8"))}
corpus = load_corpus("data/sunzi")
r3 = check_quotes(log[3]["reply"], corpus)
r4 = [(q, ok) for q, ok in check_quotes(log[4]["reply"], corpus) if not re.search(r"[A-Za-z0-9]", q)]
r3_fake = sorted({q for q, ok in r3 if not ok})
r4_ok = sum(ok for _, ok in r4)

def block(i, title):
    e = log[i]
    return f"""
**第 {i} 輪：{title}**

![design chat {i}]({RAW}design_chat_{i}.png)

<details><summary>第 {i} 輪全文（點開）</summary>

**我：** {e['prompt'].replace(chr(10), '  ' + chr(10))}

**AI（Qwen2.5-14B）：**

```
{e['reply']}
```
</details>
"""

md = f"""
## 1. 先跟生成式 AI 討論人設與「為什麼需要 RAG」（作業步驟一）

和上次作業一樣，我先開一個最陽春的聊天室（Gradio `ChatInterface`＋aisuite，模型是本機 Ollama 跑的 **Qwen2.5-14B**，程式在 repo 的 `tools/design_chat_app.py`），用四輪對話把人設從想法修到可以寫進程式的規格，順便**實測模型憑記憶引用《孫子兵法》會不會出錯**。截圖與全文都在下面；第 3、4 輪的引文我用 `tools/check_quotes.py` 和全文比對過。

- **第 1 輪：要方向。** 說明作業（RAG 機器人、老師範例是打工權益法規問答）、我想用《孫子兵法》但不想做查條文工具，請它給五個「有趣又真的需要 RAG」的方向。它給的五個（智慧軍師、歷史戰略家、策略遊戲助手、現代管理學者、兵法解讀師）都合理，但「為什麼需要 RAG」五條寫的全是同一句「需要檢索原文來找到符合的兵法原則」，沒有講到真正的理由：**模型對原文的記憶不可靠**（第 3 輪會證明）。我採用「智慧軍師」，但把對象從「商業策略」改成學生的日常煩惱（考試、報告、人際、求職），比較有趣也比較好測。
- **第 2 輪：請它寫 system prompt。** 它給了五段式格式（稱呼、引用原文、白話解讀、行動方略、結語）、「只能引用提供的段落並標註編號」、沒有相關內容時說「此處兵法未言及」。這版的洞：(1) 格式用散文描述、沒有固定小標，範例還把編號寫進引號裡「[doc_06_05]」，程式抓不到；(2) 沒要求**一字不改**，模型可能把自己的話框在「」裡，核對就失效；(3) 沒有防注入規則；(4) 沒說白話輔助可不可以當原文引用；(5) 軍師叫對方「賢弟」怪怪的，改成「主公」。
- **第 3 輪：實測「憑記憶引用」。** 不給任何段落，請它憑記憶引用原文回答「五科只夠讀三科怎麼取捨」。它很有自信地引了 {"、".join("「" + q + "」" for q in r3_fake) or "（見全文）"} 並說出自〈形篇〉——**這句《孫子兵法》裡根本沒有**（意思接近的真正原文是始計篇的「多算勝，少算不勝」）。{len(r3)} 句引文、{sum(ok for _, ok in r3)} 句在原文中。這就是這份作業要做 RAG、而且要**用程式核對引文**的理由。
- **第 4 輪：給它真正檢索到的三段原文再答一次。** 三句引文全部核對得到（{r4_ok}/{len(r4)} ✅），格式也照做了。但它把三段都整段引完、「白話解讀」變成逐句翻譯而不是對應我的處境、行動方略給了四條。所以定稿 prompt 把【兵法曰】限制在 1～2 句、【軍師解】要「對應主公的處境、3 句以內」、【方略】固定三條，並加上「一字不改」「編號放在引號外面」「白話輔助不是原文」「兵法未載」四條規則（§6）。

**最後定稿的 system prompt（§6）和 AI 第 2 輪版本的差別：** 固定四個小標【兵法曰】【軍師解】【方略】【一言】；引用必須一字不改、「」框住、編號與篇名放在方括號裡；只能引附上的原文、白話輔助不能引；兵法未載時的固定說法；防注入；稱呼改「主公」。

### 四輪對話截圖與全文
{block(1, "請 AI 給五個「有趣又真的需要 RAG」的方向")}
{block(2, "請 AI 設計「孫子 AI 軍師」的 system prompt")}
{block(3, "實測：不給段落、憑記憶引用原文 → 引了不存在的句子")}
{block(4, "給它三段真正檢索到的原文再答一次 → 三句引文都核對得到")}
"""
open("tools/sections/design.md", "w", encoding="utf-8").write(md.strip("\n") + "\n")
print("design.md written; round3 fake quotes:", r3_fake, "| round4 ok:", r4_ok, "/", len(r4))
