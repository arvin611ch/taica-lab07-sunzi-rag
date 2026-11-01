"""從維基文庫抓《孫子兵法》全文，清掉 wiki 標記，整理成 13 個 TXT（每篇一檔、我自己設計的欄位格式），再打包成 data/sunzi_source.zip。

用法：python3 tools/fetch_sunzi.py [wikitext.json]
  不給參數就直接向維基文庫 API 抓；給 JSON（API 的回應）就離線整理。
"""
import json, re, sys, os, zipfile, urllib.request, urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "sunzi"
SOURCE_URL = "https://zh.wikisource.org/wiki/%E5%AD%AB%E5%AD%90%E5%85%B5%E6%B3%95"
API = ("https://zh.wikisource.org/w/api.php?action=parse&page=%E5%AD%AB%E5%AD%90%E5%85%B5%E6%B3%95"
       "&prop=wikitext&format=json&formatversion=2")
DATE = "2026-11-01"

# 我替每一篇寫的「主旨」與「關鍵詞」：主旨是白話一句話，會和原文一起送進 embedding，
# 讓用現代話問的問題比較容易對到文言文的段落；關鍵詞只放 metadata，給介面顯示用。
CHAPTERS = [
    ("始計篇", "開戰前先把帳算清楚：用道、天、地、將、法五事與七計評估雙方勝算；兵者詭道，示弱藏強、攻其無備、出其不意。",
     ["廟算", "五事七計", "知勝負", "詭道", "出其不意", "事前評估"]),
    ("作戰篇", "打仗極度燒錢，拖越久越傷國力：兵貴勝不貴久，速戰速決；就地取用敵方資源（因糧於敵），用獎賞激勵、把俘獲變成自己的力量。",
     ["兵貴勝不貴久", "速戰速決", "因糧於敵", "成本", "賞其先得", "勝敵而益強"]),
    ("謀攻篇", "不戰而屈人之兵才是最高明：上兵伐謀、其次伐交、其次伐兵、下策攻城；依兵力對比決定圍、攻、分、守或避；知彼知己，百戰不殆。",
     ["不戰而屈人之兵", "上兵伐謀", "知彼知己", "十則圍之", "君不御將", "知勝有五"]),
    ("軍形篇", "先讓自己立於不敗之地，再等對手露出破綻：勝兵先勝而後求戰；善守者藏於九地之下；平時累積實力（度量數稱勝），勝利像積水沖下山谷一樣順理成章。",
     ["先為不可勝", "立於不敗之地", "勝兵先勝而後求戰", "攻守", "實力累積", "勝於易勝"]),
    ("兵勢篇", "用「勢」與「節」放大力量：以正合、以奇勝，奇正變化無窮；像激流漂石、張弓待發；善戰者求之於勢，不責於人，順著人的本性擺好位置。",
     ["奇正", "勢", "節", "激水漂石", "擇人而任勢", "以利動之"]),
    ("虛實篇", "主動權在我、不在敵：致人而不致於人；避實擊虛，攻其所不守、攻其所必救；讓敵人處處防備而分散，我則集中兵力；兵形象水，因敵而變。",
     ["致人而不致於人", "避實擊虛", "攻其所必救", "我專敵分", "無所不備則無所不寡", "兵形象水"]),
    ("軍爭篇", "爭取先機與有利位置：以迂為直、以患為利，不貪快而把隊伍拖垮；掌握士氣與心理的起伏（治氣、治心、治力、治變）；圍師必闕、窮寇勿迫。",
     ["以迂為直", "以患為利", "避其銳氣擊其惰歸", "治氣治心", "圍師必闕", "窮寇勿迫"]),
    ("九變篇", "靈活應變，不死守教條：有些路不走、有些城不攻、有些命令不接；智者之慮必雜於利害；將有五危（必死、必生、忿速、廉潔、愛民）會被敵人利用。",
     ["九變", "途有所不由", "雜於利害", "將有五危", "無恃其不來恃吾有以待", "彈性"]),
    ("行軍篇", "行軍、紮營與觀察敵情：在山、水、沼澤、平地各怎麼處軍；從塵土、鳥獸、使者言行判讀敵人意圖；對士兵先親近再立威，令之以文、齊之以武。",
     ["處軍", "相敵", "辭卑而益備者進也", "令之以文齊之以武", "觀察", "紀律"]),
    ("地形篇", "六種地形各有打法（通、掛、支、隘、險、遠）；六種敗仗都是將帥的錯（走、弛、陷、崩、亂、北）；知彼知己、知天知地；視卒如嬰兒但不能溺愛。",
     ["六形", "六敗", "將之至任", "視卒如嬰兒", "知天知地", "進不求名退不避罪"]),
    ("九地篇", "九種戰地的心理與戰法（散、輕、爭、交、衢、重、圮、圍、死地）：投之亡地然後存、陷之死地然後生；兵之情主速，乘人之不及；把全軍帶成一體（率然）。",
     ["九地", "投之亡地然後存", "兵之情主速", "率然", "常山之蛇", "置之死地"]),
    ("火攻篇", "用火、水等外力輔助作戰的五種方法與時機；戰勝後要能鞏固成果（費留）；主不可以怒而興師、將不可以慍而致戰，亡國不可以復存，死者不可以復生。",
     ["火攻五法", "費留", "非利不動", "怒不興師", "慍不致戰", "情緒管理"]),
    ("用間篇", "情報決定勝負：花大錢打仗卻捨不得情報費是最不仁的；五種間諜（鄉間、內間、反間、死間、生間）如何運用，反間最關鍵；先知者不可取於鬼神，必取於人。",
     ["用間", "五間", "反間", "先知", "不可取於鬼神", "情報"]),
]


def clean_wikitext(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", "", text, flags=re.S)
    text = re.sub(r"\[\[[^\]|]*\|([^\]]+)\]\]", r"\1", text)      # [[頁面|顯示]] -> 顯示
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)                # [[頁面]] -> 頁面
    text = re.sub(r"\{\{另\|([^|}]+)\|[^}]*\}\}", r"\1", text)     # {{另|甲|乙}} 異文：取正文「甲」
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)                     # 其他模板丟掉
    text = re.sub(r"'''?", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("　", "").replace(" ", "")
    return text


def main():
    if len(sys.argv) > 1:
        wikitext = json.load(open(sys.argv[1], encoding="utf-8"))["parse"]["wikitext"]
    else:
        with urllib.request.urlopen(API, timeout=30) as r:
            wikitext = json.load(r)["parse"]["wikitext"]

    sections = re.split(r"^==\s*(.+?)\s*==\s*$", wikitext, flags=re.M)
    # sections = [前言, 標題1, 內容1, 標題2, 內容2, ...]
    pairs = list(zip(sections[1::2], sections[2::2]))[:13]   # 維基文庫頁尾另有「答話」附錄，不收
    assert len(pairs) == 13 and pairs[-1][0].startswith("用間"), f"預期 13 篇：{[p[0] for p in pairs]}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.txt"):
        old.unlink()

    total_chars, rows = 0, []
    for no, ((heading, body), (name, gist, keywords)) in enumerate(zip(pairs, CHAPTERS), start=1):
        body = clean_wikitext(body)
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        paras = [re.sub(r"\s+", "", p) for p in paras]
        for p in paras:
            assert not re.search(r"[\[\]{}<>=|']", p), f"{name} 仍有 wiki 標記：{p[:40]}"
        chars = sum(len(p) for p in paras)
        total_chars += chars
        rows.append((no, name, heading, len(paras), chars, max(len(p) for p in paras)))
        content = (
            f"篇名：{name}\n"
            f"篇序：{no}\n"
            f"維基文庫標題：{heading}\n"
            f"主旨：{gist}\n"
            f"關鍵詞：{'、'.join(keywords)}\n"
            f"原典：《孫子兵法》（春秋・孫武），公有領域；文字據維基文庫整理\n"
            f"來源網址：{SOURCE_URL}\n"
            f"整理日期：{DATE}\n"
            f"\n【原文】\n" + "\n\n".join(paras) + "\n"
        )
        (OUT_DIR / f"{no:02d}_{name}.txt").write_text(content, encoding="utf-8")

    readme = f"""# 《孫子兵法》RAG 知識來源（13 篇）

- 原典：《孫子兵法》，春秋・孫武，公有領域。
- 文字來源：維基文庫 {SOURCE_URL} （整理日期 {DATE}）。清掉 wiki 標記；異文（{{{{另|甲|乙}}}}）取正文。
- 每篇一個 UTF-8 TXT，檔名 `編號_篇名.txt`。欄位：篇名、篇序、維基文庫標題、主旨（我寫的白話一句話）、關鍵詞（我整理的）、原典、來源網址、整理日期，然後是【原文】；原文段落之間空一行。
- 第一份程式（RAG01）只把「主旨」與「原文段落」送進 embedding；關鍵詞、網址等只當 metadata。

| 篇序 | 篇名 | 段落數 | 字數 | 最長段落 |
|---|---|---|---|---|
""" + "\n".join(f"| {no} | {name} | {n} | {c} | {m} |" for no, name, _, n, c, m in rows) + f"\n\n合計 {total_chars} 字。\n"
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    zip_path = ROOT / "data" / "sunzi_source.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT_DIR.iterdir()):
            z.write(p, arcname=p.name)
    for no, name, heading, n, c, m in rows:
        print(f"{no:2d} {name} ({heading})  段落 {n:2d}  字 {c:4d}  最長段 {m}")
    print(f"合計 {total_chars} 字 → {zip_path} ({zip_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
