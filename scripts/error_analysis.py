"""Phân tích lỗi mục 2.5: chọn 20 dòng sai (seed 42) và phân loại tự động thô.

Dùng: python scripts/error_analysis.py results/ft_4epochs/pred_ppocrv5_mobile_4epochs.jsonl
Nhóm tự động: ii = chỉ khác khoảng trắng; i = khác dấu/thanh nhưng cùng chữ cái gốc;
iii = còn lại. Bảng 20 dòng trong docs/error_analysis_2_5.md được xếp tay lại sau khi đọc từng dòng.
"""
import json, random, sys, unicodedata as u
from collections import Counter

def base(s):
    s = u.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if not u.combining(c))

def cls(g, p):
    g, p = u.normalize("NFC", g), u.normalize("NFC", p)
    if g == p: return None
    if g.replace(" ", "") == p.replace(" ", ""): return "ii"
    if base(g).replace(" ", "") == base(p).replace(" ", ""): return "i"
    return "iii"

rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
wrong = [(r, cls(r["gt"], r["pred"])) for r in rows if cls(r["gt"], r["pred"])]
c = Counter(k for _, k in wrong)
print(f"{len(wrong)}/{len(rows)} dòng sai;", {k: f"{100*v/len(wrong):.1f}%" for k, v in sorted(c.items())})
random.seed(42)
for i, (r, k) in enumerate(random.sample(wrong, 20), 1):
    print(i, k, r["image"].split("/")[-1], "\n  GT :", r["gt"], "\n  PRD:", r["pred"])
