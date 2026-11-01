"""引文核對：把一段回答裡用「」框起來的句子，拿去和《孫子兵法》全文比對，看是不是真的原文。
用法：python tools/check_quotes.py data/sunzi "<回答文字>"  或  python tools/check_quotes.py data/sunzi --design-log screenshots/design_chat_log.json 3
"""
import re, sys, json
from pathlib import Path

PUNCT = r"[，。；：、！？「」『』（）()\s]"


def load_corpus(data_dir):
    text = ""
    for p in sorted(Path(data_dir).glob("*.txt")):
        text += p.read_text(encoding="utf-8").split("【原文】", 1)[1].split("【白話輔助", 1)[0]
    return re.sub(PUNCT, "", text)


def check_quotes(answer, corpus_nopunct, min_len=4):
    """回傳 [(引文, 是否在原文中)]；比對時忽略標點與空白。"""
    results = []
    for q in re.findall(r"「([^「」]+)」", answer):
        core = re.sub(PUNCT, "", q)
        if len(core) < min_len:
            continue
        results.append((q, core in corpus_nopunct))
    return results


if __name__ == "__main__":
    corpus = load_corpus(sys.argv[1])
    if sys.argv[2] == "--design-log":
        log = json.load(open(sys.argv[3], encoding="utf-8"))
        answer = next(e for e in log if e["round"] == int(sys.argv[4]))["reply"]
    else:
        answer = sys.argv[2]
    res = check_quotes(answer, corpus)
    for q, ok in res:
        print(("✅ 原文有" if ok else "❌ 原文沒有"), q)
    print(f"共 {len(res)} 句引文，{sum(ok for _, ok in res)} 句在原文中")
