"""把說明文字塞進「已執行」的 notebook，不用重新執行程式格：
1. 依 metadata.id 把所有 markdown 格的內容同步成 *.unexecuted.ipynb（build_notebooks.py 產生）裡的最新版本；
2. 再把 tools/sections/*.md 塞進對應的佔位 markdown 格。
用法：python tools/patch_markdown.py RAG02_sunzi_strategist_chatbot.ipynb
"""
import sys, nbformat
NB = sys.argv[1] if len(sys.argv) > 1 else "RAG02_sunzi_strategist_chatbot.ipynb"
UNEXEC = NB.replace(".ipynb", ".unexecuted.ipynb")
PLACEHOLDERS = {"design": "__DESIGN_SECTION__", "shots": "__SHOTS_SECTION__",
                "ai-help": "__AIHELP_SECTION__", "reflection": "__REFLECTION_SECTION__"}
nb = nbformat.read(NB, as_version=4)
try:
    latest = {c.metadata.get("id"): c.source for c in nbformat.read(UNEXEC, as_version=4).cells if c.cell_type == "markdown"}
except FileNotFoundError:
    latest = {}
synced, done = [], []
for cell in nb.cells:
    cid = cell.metadata.get("id")
    if cell.cell_type != "markdown":
        continue
    if cid in latest and cell.source != latest[cid] and cid not in PLACEHOLDERS:
        cell.source = latest[cid]; synced.append(cid)
    if cid in PLACEHOLDERS:
        try:
            cell.source = open(f"tools/sections/{cid}.md", encoding="utf-8").read().strip("\n"); done.append(cid)
        except FileNotFoundError:
            print("missing section file for", cid)
nbformat.write(nb, NB)
left = [c.metadata.get("id") for c in nb.cells if c.cell_type == "markdown" and "_SECTION__" in c.source]
print("synced from builder:", synced, "| sections patched:", done, "| placeholders left:", left)
