# 繳交文字（貼到課程平台）

> **繳交標題：** 陽明交大 113611097 百川 江晟維 孫子 AI 軍師（RAG）
>
> **重點截圖請上傳：** `screenshots/app_2_answer.png`（Gradio 問答＋引文核對＋檢索原文）、`screenshots/app_5_offtopic.png`（兵法未載）、`screenshots/design_chat_3.png`（AI 憑記憶引用出不存在的句子）。若平台可多傳，再加 `app_4_rewrite.png`（翻成兵法語再檢索）與 `design_chat_4.png`。

---

**Colab 連結（第二份程式 RAG02，任何人可開啟）：**
https://colab.research.google.com/github/arvin611ch/taica-lab07-sunzi-rag/blob/main/RAG02_sunzi_strategist_chatbot.ipynb

第一份程式（RAG01，建向量資料庫）：
https://colab.research.google.com/github/arvin611ch/taica-lab07-sunzi-rag/blob/main/RAG01_build_sunzi_vector_db.ipynb

rag-data.zip（程式二會自動下載）：https://raw.githubusercontent.com/arvin611ch/taica-lab07-sunzi-rag/main/rag-data.zip
原始碼、資料、全部截圖與 AI 協作紀錄：https://github.com/arvin611ch/taica-lab07-sunzi-rag

**我所使用的資料：**
《孫子兵法》十三篇全文（春秋・孫武，公有領域；文字據維基文庫整理，7,434 字）。每篇一個 TXT，欄位是我自己設計的：篇名、篇序、主旨（我寫的白話一句）、關鍵詞、來源網址、【原文】（段落空行隔開）、【白話輔助】（用本機 Qwen2.5-14B 逐段翻成白話、人工抽查，存進 metadata 給軍師理解用）。程式一把 13 篇切成 74 段、一段一向量（不是老師的一篇一向量），先做小實驗比較「只放原文／加篇名主旨／再加白話輔助」三種寫法對白話問題的檢索命中率（16 題的 hit@3 分別 8／10／9，白話輔助反而沒幫助），程式自動用最好的「篇名＋主旨」建 FAISS 索引，匯出 `rag-data.zip`。

**人設／背景設定：**
「孫子 AI 軍師」：以《孫子兵法》作者孫武的口吻替現代人出謀劃策，稱使用者「主公」，把考試、報告、人際、求職的煩惱當成一場戰役分析。回答固定四段：【兵法曰】只引用程式檢索到的原文（一字不改、「」框住、方括號標段落編號與篇名）→【軍師解】用台灣白話把原文對應到主公的處境→【方略】三條具體行動→【一言】一句話收尾。兵法沒談到的事（例如晚餐吃什麼）要說「兵法未載此事」、不硬套。每一句引文與編號都由**程式**核對（去標點後在 74 段原文裡找子字串），不是請模型自己說有沒有引對。另有多輪對話（`gr.State`）與可選的「先把問題翻成兵法語再檢索」。

**重點說明（Notebook 內以 Markdown 標示）：**
- §1 先和生成式 AI（本機 Qwen2.5-14B）四輪對話討論人設：要方向、請它寫 system prompt、**實測它憑記憶引用《孫子兵法》會引出不存在的句子**（「故數戰則勝，無數則不勝」）、給它真正檢索到的原文後三句引文都核對得到；附四張截圖（含 prompt 與回覆）與全文，並列出定稿 prompt 與 AI 版本的差異。
- §2 從 GitHub 網址自動下載 `rag-data.zip`（助教不用準備檔案；也支援 Google Drive 連結）。
- §3 沿用老師的檢查邏輯驗證 zip 內四個檔案，並顯示程式一的小實驗結果。
- §4 金鑰：Colab 🔑 依序找 Groq／OpenRouter／OpenAI／Anthropic；本機自動改用 Ollama。
- §5 檢索函式，以及選用的「先翻成兵法語再檢索」（HyDE）並示範差異。
- §6 軍師 system prompt 與多輪訊息組法，每條規則都說明為什麼要有。
- §7 `verify_quotes()`／`verify_citations()`：程式核對引文與編號。
- §8 文字版測試：完整走一次、**純 LLM vs RAG 引文核對（pandas／matplotlib 畫圖）**、兵法未載、多輪追問。
- §9 Gradio Blocks：聊天視窗、top-k、改寫開關、核對結果、可收合的檢索原文。
- §10 Gradio 對話結果截圖（實際操作：問答 → 追問 → 開改寫再問 → 兵法未載）。
- §11 依註三逐項說明 AI 協助的地方與理解後的說明。
- §12 心得：文件怎麼寫比 prompt 更影響檢索、模型憑記憶引古文錯得很有自信、一段一向量的取捨、改寫檢索是雙面刃、多輪檢索的弱點。

**有請生成式 AI 幫忙的地方：** 見 Notebook §1、§11 與 repo 的 `ai_assist_log.md`（含 prompt 與生成結果截圖）。
