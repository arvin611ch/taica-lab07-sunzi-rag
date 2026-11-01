"""替 data/sunzi/*.txt 的每個原文段落產生「白話輔助文」（本機 Ollama Qwen2.5-14B），存成 tools/paraphrases.json，
再寫回每篇 TXT 的【白話輔助】區段（編號對應原文段落）。白話文只拿來幫助檢索；回答時軍師只能引用原文。

用法：python tools/make_paraphrases.py            # 產生（可中斷續跑）＋寫回 TXT
      python tools/make_paraphrases.py --write-only  # 只把現有 JSON 寫回 TXT
"""
import json, re, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "sunzi"
OUT = ROOT / "tools" / "paraphrases.json"
MODEL = "ollama:qwen2.5:14b"

PROMPT = """你是教《孫子兵法》的國文老師。請把下面這段《{chapter}》的原文翻成台灣習慣的白話文。
要求：1～3 句、80 字以內；忠於原文、不要加評論、不要加「這段話的意思是」之類的前綴；用現代人聽得懂的詞（例如「軍隊」「敵人」「將領」）。只輸出白話翻譯本身。

原文：{text}"""


def load_chapters():
    chapters = []
    for p in sorted(DATA.glob("*.txt")):
        raw = p.read_text(encoding="utf-8")
        head, rest = raw.split("【原文】", 1)
        body = rest.split("【白話輔助", 1)[0]
        field = lambda k: re.search(rf"^{k}：(.*)$", head, re.M).group(1).strip()
        paras = [x.strip() for x in re.split(r"\n\s*\n", body) if x.strip()]
        chapters.append({"path": p, "head": head, "no": int(field("篇序")), "name": field("篇名"), "paras": paras})
    return chapters


def generate(chapters):
    import aisuite as ai
    client = ai.Client(provider_configs={"ollama": {"timeout": 600}})
    done = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    todo = [(c, i, t) for c in chapters for i, t in enumerate(c["paras"], 1) if f"doc_{c['no']:02d}_{i:02d}" not in done]
    print(f"已完成 {len(done)}，待產生 {len(todo)}", flush=True)
    for c, i, text in todo:
        doc_id = f"doc_{c['no']:02d}_{i:02d}"
        t0 = time.time()
        for attempt in range(3):
            r = client.chat.completions.create(model=MODEL, temperature=0.2,
                                               messages=[{"role": "user", "content": PROMPT.format(chapter=c["name"], text=text)}])
            out = r.choices[0].message.content.strip().replace("\n", " ")
            out = re.sub(r"^(白話(翻譯|文)?[：:]\s*)", "", out)
            if 10 <= len(out) <= 160 and "原文" not in out[:4]:
                break
        done[doc_id] = out
        OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{doc_id} ({time.time() - t0:4.1f}s) {out[:60]}", flush=True)
    return done


def write_back(chapters, para):
    for c in chapters:
        lines = []
        for i, _ in enumerate(c["paras"], 1):
            key = f"doc_{c['no']:02d}_{i:02d}"
            if key not in para:
                raise SystemExit(f"缺 {key} 的白話文，先跑產生步驟")
            lines.append(f"{i}. {para[key]}")
        content = (c["head"] + "【原文】\n" + "\n\n".join(c["paras"]) + "\n\n"
                   "【白話輔助】（由 AI 模型 Qwen2.5-14B 逐段翻成白話、人工抽查；給軍師當理解輔助、也用於檢索實驗，回答時只引用原文）\n"
                   + "\n".join(lines) + "\n")
        c["path"].write_text(content, encoding="utf-8")
    readme = DATA / "README.md"
    txt = readme.read_text(encoding="utf-8")
    note = "- 每篇末尾另有【白話輔助】：各段原文的白話翻譯（AI 產生、人工抽查），編號對應原文段落；給第二本程式的軍師當理解輔助、也用於第一本的檢索實驗，不會被當成原文引用。\n"
    if "【白話輔助】" not in txt:
        txt = txt.replace("- 第一份程式（RAG01）", note + "- 第一份程式（RAG01）")
        readme.write_text(txt, encoding="utf-8")
    zip_path = ROOT / "data" / "sunzi_source.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DATA.iterdir()):
            z.write(p, arcname=p.name)
    print(f"寫回 {len(chapters)} 篇；重新打包 {zip_path} ({zip_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    chapters = load_chapters()
    para = json.load(open(OUT, encoding="utf-8")) if "--write-only" in sys.argv else generate(chapters)
    write_back(chapters, para)
