# AI 協作紀錄（依作業註三）

本作業在 **Claude Code**（Anthropic Claude，終端機裡的程式協作工具）協作下完成；人設討論、憑記憶引用的實測、以及每段原文的白話輔助文，另外用本機 Ollama 的 **Qwen2.5-14B**（四輪對話的截圖與全文在 RAG02 notebook §1 與 `screenshots/design_chat_*.png`、`screenshots/design_chat_log.json`）。

## 給 Claude Code 的要求

1. 在 `taica_gen_ai/` 建立 `lab07`，貼上第七週作業說明（打造自己的 RAG 對話機器人：RAG01 建向量資料庫存成 rag-data.zip、RAG02 讀 zip 網址做 RAG 對話機器人、Gradio 展示、繳交內容、評分標準、註一至註四），要求完成整份作業。
2. 一切照作業規定：保留老師固定套件、主題不能跟老師範例相似、程式要能從 zip 網址讀資料讓助教執行、有用 AI 的地方要說明並附 prompt 與結果截圖。

## Claude Code 做的事（依時間順序）

1. 下載並閱讀老師這學期的 `RAG_01_打造你的_RAG_系統.ipynb`、`RAG_02_使用篇.ipynb` 與 `rag-data.zip`（yenlung/Generative-AI），以及舊版 Demo06 系列，整理出固定套件、EmbeddingGemma 2＋FAISS 的建庫與匯出格式、aisuite 與 Gradio 寫法。
2. 決定主題：《孫子兵法》全文（公有領域）＋「孫子 AI 軍師」人設，刻意避開「法規／校園資料問答」類型；從維基文庫抓全文、清 wiki 標記、每篇一檔（`tools/fetch_sunzi.py`），並替 13 篇各寫一句主旨與關鍵詞。
3. 開最陽春的聊天室 `tools/design_chat_app.py`，用 Playwright（`tools/shoot_design_chat.py`）送出四輪提問給 Qwen2.5-14B 並截圖；用 `tools/check_quotes.py` 核對第 3、4 輪的引文（第 3 輪引了不存在的「故數戰則勝，無數則不勝」）。
4. 先做一個檢索品質小實驗（16 個白話問題 × 三種文件寫法），發現只放原文命中率很低，決定加入「主旨」並試試「白話輔助」；白話輔助用 `tools/make_paraphrases.py` 請 Qwen2.5-14B 逐段產生，寫回每篇 TXT 的【白話輔助】區段（實驗結果見下方「測試中發現」）。
5. 撰寫 `tools/build_notebooks.py`，一次產生兩本 notebook 與 `app.py`：RAG01（切段、實驗、建庫、匯出 `rag-data.zip`）、RAG02（從 GitHub 網址下載 zip、沿用老師的檢查、軍師 prompt、引文核對、純 LLM vs RAG、兵法未載、多輪、Gradio Blocks）。
6. 用 nbclient 從頭執行兩本 notebook 產生所有輸出；啟動 `app.py`，用 Playwright（`tools/shoot_app.py`）實際問答、追問、開啟改寫檢索、問兵法未載的問題並截圖。
7. 建立 GitHub repo、推上 notebook、`rag-data.zip`、資料與截圖，產生 Colab 連結與繳交文字。

## 測試中發現並修正的事

- 本機沒有雲端 API 金鑰；同機共用的 Ollama（port 11434）因 GPU 被其他程式佔滿、大多跑在 CPU 上，每次呼叫 20～100 秒，後來發現同機另一個實例（port 11435）在 GPU 上，改設 `OLLAMA_API_URL` 指過去，每次呼叫降到 1～3 秒。Colab 版沿用老師的 Groq，本機只拿來驗證流程與截圖。
- EmbeddingGemma 2 在本機載入時要求 torchvision（雖然只用文字部分），補裝即可；Colab 內建就有。
- 維基文庫的《孫子兵法》頁面尾端多一節「答話」附錄，抓取時只取前十三篇。
- 純文言原文對白話問題的檢索命中率低（16 題 hit@3 8），加篇名主旨 10，再加白話輔助反而 9；程式一的實驗格自動採用「篇名＋主旨」建庫，白話輔助只存進 metadata 給軍師理解用（數字見 RAG01 §8）。另外用同樣 16 題測「先改寫成兵法語再檢索」：hit@1 7→8、hit@3 10→9，等於沒差，所以做成預設關閉的開關。
- 引文核對發現「兵無常勢，水無常形」被打 ❌，查證後是版本差異（維基文庫這一版作「兵無成勢，無恒形」），於是在 §8.2 與心得說明「❌ 不一定是捏造」。
- 引文核對一開始把「孫子 AI 軍師」這種「」框起來的名稱也當引文，改成跳過含英數字的字串。
- 第 4 輪實測發現模型會把三段原文整段引完、方略給四條，定稿 prompt 因此限制【兵法曰】1～2 句、【方略】固定三條。
