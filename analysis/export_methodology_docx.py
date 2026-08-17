# -*- coding: utf-8 -*-
"""
将「动态基线算法方法论文档.md」转换为学术论文排版的 Word 文档。

- 标题/小节 → Word 标题样式（黑体加粗）
- 正文 → 宋体 11pt，1.5 倍行距
- 行间公式 \\[ ... \\] → matplotlib mathtext 渲染为图片，右对齐编号 (1)…
- 行内公式 \\( ... \\) → Word 原生上下标/斜体 run（Cambria）
- 表格 → Table Grid 样式；图片 → 居中插入 + 图注

运行: python -m analysis.export_methodology_docx
输出: 动态基线算法方法论文档.docx（项目根目录）
"""
from __future__ import annotations
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml.ns import qn

from .config import ROOT, OUT_INTER

MD_PATH = ROOT / "动态基线算法方法论文档.md"
DOCX_PATH = ROOT / "动态基线算法方法论文档.docx"
EQ_DIR = OUT_INTER / "docx_eq"
EQ_DIR.mkdir(parents=True, exist_ok=True)

EQ_DPI = 200

# 行间公式（按文档出现顺序）的 mathtext 表达式
CURATED_EQS = [
    r"\frac{\Delta R}{R_0}(t_i) = \frac{R_{\mathrm{clean}}(t_i) - R_0(t_i)}{R_0(t_i)}",
    r"\frac{\Delta R}{R_0}(t_i) \approx \frac{D(t_i)}{R_0(t_i)}, \quad D(t_i) := R_{\mathrm{clean}}(t_i) - R_0(t_i)",
    r"\mathrm{MAD}(x) = \mathrm{median}(|x_i - \mathrm{median}(x)|)",
    r"f_s = 5.0\ \mathrm{Hz}, \qquad t_k = t_0 + k / f_s",
    r"R(t_i) > 1.5 \times \mathrm{median}(R) \quad \mathrm{or} \quad R(t_i) < \mathrm{median}(R) / 1.5",
    r"|R(t_i) - \tilde{R}(t_i)| > k_{\mathrm{out}} \cdot \max(\mathrm{MAD}_R(t_i),\ 0.5\,\mathrm{MAD}_{\mathrm{global}})",
    r"R_0(t_i) = Q_q\{ R_{\mathrm{clean}}(t_j) : |t_j - t_i| \leq W_b/2 \}",
    r"\frac{\Delta R}{R_0}(t_i) = \frac{R_{\mathrm{clean}}(t_i) - R_0(t_i)}{R_0(t_i)}, \quad |R_0(t_i)| > 10^{-6}",
    r"\delta^{*} = \mathrm{argmin}_{\delta} \sum_{w} \left| N_{\mathrm{peak}}^{(w)}(\delta) - C^{(w)} \right|",
    r"\mathrm{pad} = \min\left(1.0,\ \frac{1.0}{\max(1,\ T_w / 2)}\right)\ \mathrm{s}",
    r"\frac{\Delta R}{R_0} > \mathrm{median} + k_w \cdot \max(1.4826 \times \mathrm{MAD},\ \sigma_{\min})",
]

# ---------- 行内公式 LaTeX → Word run ----------
MACRO_MAP = {
    r"\Delta": "Δ", r"\delta": "δ", r"\sigma": "σ", r"\times": "×",
    r"\le": "≤", r"\ge": "≥", r"\approx": "≈", r"\to": "→",
    r"\in": "∈", r"\pm": "±", r"\cdot": "·", r"\mid": "|",
    r"\sum": "∑", r"\min": "min", r"\max": "max", r"\arg": "arg",
    r"\,": " ", r"\;": " ", r"\quad": "  ", r"\qquad": "    ",
    r"\left": "", r"\right": "", r"\big": "", r"\|": "|",
    r"\{": "{", r"\}": "}", r"\%": "%",
}


def _read_group(s: str, i: int):
    """读取 {..} 组（i 指向 '{'），返回 (内容, 下一索引)。"""
    assert s[i] == "{"
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _parse_inline(s: str, upright=False, script=None):
    """递归解析行内 LaTeX，返回 [(text, italic, script)] 扁平 run 列表。"""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            # 非字母宏（\, \; \{ \} \| \%）
            if i + 1 < n and s[i + 1] in ",;{}|%":
                out.append((MACRO_MAP["\\" + s[i + 1]], False, script))
                i += 2
                continue
            # 宏
            m = re.match(r"\\([a-zA-Z]+)", s[i:])
            if m:
                name = m.group(0)
                i += len(name)
                if name in (r"\mathrm", r"\text", r"\operatorname"):
                    grp, i = _read_group(s, i)
                    out.extend(_parse_inline(grp, upright=True, script=script))
                    continue
                if name == r"\frac":
                    a, i = _read_group(s, i)
                    b, i = _read_group(s, i)
                    out.extend(_parse_inline(a, upright, script))
                    out.append(("/", False, script))
                    out.extend(_parse_inline(b, upright, script))
                    continue
                if name == r"\tilde":
                    grp, i = _read_group(s, i)
                    sub = _parse_inline(grp, upright, script)
                    if sub:
                        t, it, sc = sub[0]
                        out.append((t + "̃", it, sc))
                        out.extend(sub[1:])
                    continue
                out.append((MACRO_MAP.get(name, name.lstrip("\\")), False, script))
                continue
            out.append(("\\", False, script))
            i += 1
            continue
        if c == "_" or c == "^":
            kind = "sub" if c == "_" else "sup"
            i += 1
            if i < n and s[i] == "{":
                grp, i = _read_group(s, i)
            else:
                grp = s[i] if i < n else ""
                i += 1
            out.extend(_parse_inline(grp, upright, script=kind))
            continue
        if c == "{":
            grp, i = _read_group(s, i)
            out.extend(_parse_inline(grp, upright, script))
            continue
        # 普通字符
        italic = (not upright) and c.isalpha()
        out.append((c, italic, script))
        i += 1
    return out


def add_math_runs(par, latex, size=10.5):
    """把行内公式以 Cambria 斜体/上下标 run 追加到段落。"""
    for text, italic, script in _parse_inline(latex):
        if not text:
            continue
        run = par.add_run(text)
        run.font.name = "Cambria"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria")
        run.font.size = Pt(size)
        run.font.italic = bool(italic)
        if script == "sub":
            run.font.subscript = True
        elif script == "sup":
            run.font.superscript = True


# ---------- 行内混排（粗体/代码/公式） ----------
INLINE_RE = re.compile(
    r"(\*\*[^*]+\*\*|`[^`]+`|\\\(.+?\\\)|\*[^*\n]+\*)")


def add_rich_paragraph(doc, text, style=None, italic_all=False,
                       size=11, bold_all=False):
    par = doc.add_paragraph(style=style)
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            _plain_run(par, text[pos:m.start()], size, italic_all, bold_all)
        tok = m.group(0)
        if tok.startswith("**"):
            r = _plain_run(par, tok[2:-2], size, italic_all, True)
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1])
            r.font.name = "Consolas"
            r._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            r.font.size = Pt(size - 1)
        elif tok.startswith("\\("):
            add_math_runs(par, tok[2:-2], size)
        elif tok.startswith("*"):
            r = _plain_run(par, tok[1:-1], size, True, bold_all)
        pos = m.end()
    if pos < len(text):
        _plain_run(par, text[pos:], size, italic_all, bold_all)
    return par


def _plain_run(par, text, size, italic, bold):
    r = par.add_run(text)
    r.font.name = "Times New Roman"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    r.font.size = Pt(size)
    r.font.italic = italic
    r.font.bold = bold
    return r


# ---------- 行间公式图片 ----------
def render_eq_image(idx: int, mtx: str) -> Path:
    out = EQ_DIR / f"eq_{idx:02d}.png"
    fig = plt.figure(figsize=(8, 1))
    fig.text(0.0, 0.5, f"${mtx}$", fontsize=14, va="center")
    fig.savefig(out, dpi=EQ_DPI, bbox_inches="tight", pad_inches=0.08,
                facecolor="white")
    plt.close(fig)
    return out


def add_equation_paragraph(doc, idx: int, mtx: str):
    img = render_eq_image(idx, mtx)
    w_px, h_px = Image.open(img).size
    w_in = min(6.2, w_px / EQ_DPI)
    par = doc.add_paragraph()
    fmt = par.paragraph_format
    fmt.space_before = Pt(6)
    fmt.space_after = Pt(6)
    fmt.tab_stops.add_tab_stop(Inches(6.6), WD_TAB_ALIGNMENT.RIGHT)
    run = par.add_run()
    run.add_picture(str(img), width=Inches(w_in))
    r = par.add_run(f"\t({idx})")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)


# ---------- 表格 ----------
def flush_table(doc, rows):
    if len(rows) < 2:
        return
    header, body = rows[0], rows[1:]
    tbl = doc.add_table(rows=1 + len(body), cols=len(header))
    tbl.style = "Table Grid"
    for j, cell_text in enumerate(header):
        _cell_set(tbl.rows[0].cells[j], cell_text, bold=True)
    for i, row in enumerate(body):
        for j in range(len(header)):
            txt = row[j] if j < len(row) else ""
            _cell_set(tbl.rows[i + 1].cells[j], txt, bold=False)
    doc.add_paragraph()


def _cell_set(cell, text, bold):
    cell.text = ""
    par = cell.paragraphs[0]
    add_rich_paragraph_into(par, text, size=9, bold_all=bold)


def add_rich_paragraph_into(par, text, size=9, bold_all=False):
    pos = 0
    for m in INLINE_RE.finditer(text):
        if m.start() > pos:
            _plain_run(par, text[pos:m.start()], size, False, bold_all)
        tok = m.group(0)
        if tok.startswith("**"):
            _plain_run(par, tok[2:-2], size, False, True)
        elif tok.startswith("\\("):
            add_math_runs(par, tok[2:-2], size)
        elif tok.startswith("`"):
            _plain_run(par, tok[1:-1], size, False, bold_all)
        elif tok.startswith("*"):
            _plain_run(par, tok[1:-1], size, True, bold_all)
        pos = m.end()
    if pos < len(text):
        _plain_run(par, text[pos:], size, False, bold_all)


def _split_row(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


# ---------- 主转换 ----------
def convert():
    lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    doc = Document()

    # 默认样式
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    for lvl, (style_name, size) in enumerate([("Heading 1", 15),
                                              ("Heading 2", 13),
                                              ("Heading 3", 12)], start=1):
        st = doc.styles[style_name]
        st.font.name = "Times New Roman"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = RGBColor(0, 0, 0)
        st._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")

    eq_no = 0
    table_buf = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        s = line.strip()

        # 表格
        if s.startswith("|"):
            table_buf.append(_split_row(s))
            i += 1
            continue
        if table_buf:
            rows = [r for r in table_buf
                    if not all(set(c) <= set("-: ") for c in r)]
            flush_table(doc, rows)
            table_buf = []

        if not s or s == "---":
            i += 1
            continue

        if s.startswith("# "):
            p = doc.add_heading(s[2:], level=0)
            for r in p.runs:
                r.font.color.rgb = RGBColor(0, 0, 0)
        elif s.startswith("### "):
            doc.add_heading(s[4:], level=3)
        elif s.startswith("## "):
            doc.add_heading(s[3:], level=2)
        elif s.startswith("> "):
            add_rich_paragraph(doc, s[2:], italic_all=True, size=10)
        elif s.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", s)
            if m:
                alt, rel = m.group(1), m.group(2)
                img_path = ROOT / rel
                if img_path.exists():
                    par = doc.add_paragraph()
                    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    run = par.add_run()
                    run.add_picture(str(img_path), width=Inches(6.3))
                    cap = doc.add_paragraph()
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cr = cap.add_run(alt)
                    cr.font.size = Pt(9.5)
                    cr.font.name = "Times New Roman"
                    cr._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        elif s.startswith("\\["):
            buf = s[2:]
            while "\\]" not in buf and i + 1 < n:
                i += 1
                buf += " " + lines[i].strip()
            mtx = buf.split("\\]")[0].strip()
            eq_no += 1
            if eq_no <= len(CURATED_EQS):
                add_equation_paragraph(doc, eq_no, CURATED_EQS[eq_no - 1])
            else:
                add_rich_paragraph(doc, mtx)
        elif re.match(r"^- ", s) or re.match(r"^\d+\. ", s):
            body = re.sub(r"^(- |\d+\. )", "", s)
            add_rich_paragraph(doc, "• " + body if s.startswith("-") else body,
                               size=10.5)
        elif s.startswith("*") and s.endswith("*") and not s.startswith("**"):
            add_rich_paragraph(doc, s[1:-1], italic_all=True, size=10)
        else:
            add_rich_paragraph(doc, s)
        i += 1

    if table_buf:
        rows = [r for r in table_buf
                if not all(set(c) <= set("-: ") for c in r)]
        flush_table(doc, rows)

    if eq_no != len(CURATED_EQS):
        print(f"[WARN] 行间公式数量 {eq_no} 与 CURATED_EQS {len(CURATED_EQS)} 不一致")

    doc.save(str(DOCX_PATH))
    print(f"已保存: {DOCX_PATH}（共 {eq_no} 个编号公式）")


if __name__ == "__main__":
    convert()
