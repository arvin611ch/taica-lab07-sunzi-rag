# TAICA 生成式 AI｜Lab 07：用《孫子兵法》打造 RAG 對話機器人「孫子 AI 軍師」

陽明交大 113611097 百川 江晟維

- **程式二（RAG02，對話機器人；繳交用）Colab：** https://colab.research.google.com/github/arvin611ch/taica-lab07-sunzi-rag/blob/main/RAG02_sunzi_strategist_chatbot.ipynb
- **程式一（RAG01，建向量資料庫）Colab：** https://colab.research.google.com/github/arvin611ch/taica-lab07-sunzi-rag/blob/main/RAG01_build_sunzi_vector_db.ipynb
- **向量資料庫：** `rag-data.zip`（程式二會自動從 https://github.com/arvin611ch/taica-lab07-sunzi-rag/raw/main/rag-data.zip 下載）
- **知識來源：** `data/sunzi/`（13 篇 TXT）與 `data/sunzi_source.zip`
- **截圖：** `screenshots/`（`design_chat_*.png` 是和 AI 討論人設的四輪對話；`app_*.png` 是 Gradio 實際操作）
- **AI 協作紀錄：** `ai_assist_log.md`

## 作品是什麼

知識來源是《孫子兵法》十三篇全文（公有領域，文字據維基文庫整理，7,434 字）。程式一把每篇切成段落（共 74 段），先用 16 個白話問題做小實驗比較三種文件寫法（只放原文／加篇名＋主旨／再加 AI 翻的白話輔助，hit@3 分別 8／10／9），自動用最好的「篇名＋主旨」寫法以 EmbeddingGemma 2 向量化，建 FAISS 索引，匯出 `rag-data.zip`；白話輔助存進 metadata 給軍師理解用。程式二下載 zip，做一個「孫子 AI 軍師」：使用者把考試、報告、人際、求職的煩惱當成一場戰役來問，軍師只引用《孫子兵法》真正的原文（**每一句引文都由程式和全文比對**），再用台灣白話解讀、給三條行動方略。

和老師範例（打工權益法規問答）不同的地方：一段一向量而不是一篇一向量、文件寫法做過實驗、人設與固定四段格式、引文核對、純 LLM vs RAG 對照、多輪對話、可選的「先把問題翻成兵法語再檢索」。

## 怎麼跑

**Colab（助教請用這個）**：開上面程式二的連結 → 左側 🔑 新增金鑰（名稱 `Groq`；也可用 `OpenRouter`／`OpenAI`／`Anthropic`）並打開筆記本存取權 → 全部執行。`rag-data.zip` 會自動下載，最後一格開 Gradio 網頁。程式一不需要金鑰，直接全部執行即可（文本 zip 也會自動下載）。

**本機**（需要 Python 3.12；沒有雲端金鑰時會自動改用本機 Ollama 的 `qwen2.5:14b`）：

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python "sentence-transformers==6.1.0" "transformers==5.20.1" "faiss-cpu==1.15.1" \
    "aisuite[openai,groq,anthropic]==0.2.0" "groq==0.37.1" "gradio==6.30.0" torchvision numpy pandas matplotlib seaborn
export GROQ_API_KEY=...        # 或不設定，改用本機 Ollama
.venv/bin/python app.py        # http://127.0.0.1:7863
```

`app.py` 是由 `tools/build_notebooks.py` 從程式二的程式格產生的，兩邊程式一致。

## 檔案

| 檔案 | 用途 |
|---|---|
| `RAG01_build_sunzi_vector_db.ipynb` | 程式一：讀 13 篇 TXT → 切段 → 三種寫法實驗 → EmbeddingGemma 2 → FAISS → `rag-data.zip`（已在本機執行過，輸出都在裡面） |
| `RAG02_sunzi_strategist_chatbot.ipynb` | 程式二：下載 zip → 檢索 → 軍師回答 → 引文核對 → Gradio（已在本機用 Ollama Qwen2.5-14B 執行過一次） |
| `rag-data.zip` | 程式一的輸出：`index.faiss`、`documents.json`、`index_doc_ids.json`、`config.json` |
| `app.py` | 獨立可跑的 Gradio 版本 |
| `data/sunzi/*.txt`、`data/sunzi_source.zip` | 知識來源（格式見 `data/sunzi/README.md`） |
| `tools/fetch_sunzi.py` | 從維基文庫抓全文、清標記、每篇一檔（含我寫的主旨與關鍵詞） |
| `tools/make_paraphrases.py`、`tools/paraphrases.json` | 用 Qwen2.5-14B 逐段產生白話輔助並寫回 TXT |
| `tools/build_notebooks.py` | 產生兩本 notebook 與 `app.py`（程式的唯一來源） |
| `tools/check_quotes.py` | 引文核對工具（比對「」裡的句子是否真的在原文中） |
| `tools/design_chat_app.py`、`tools/shoot_design_chat.py`、`tools/design_prompts.json` | 和 AI 討論人設用的聊天室與自動截圖 |
| `tools/shoot_app.py` | 用 Playwright 實際操作 Gradio 並截圖 |
| `tools/make_design_section.py`、`tools/make_shots_section.py`、`tools/patch_markdown.py`、`tools/sections/` | 產生並塞入 notebook 的說明段落 |
| `screenshots/*.png`、`screenshots/*.json` | 截圖與對話全文 |
| `ai_assist_log.md` | 依作業註三說明 AI 協作內容 |
| `submission.md` | 繳交文字 |
