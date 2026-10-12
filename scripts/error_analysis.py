"""Mục 2.5 (đề bài): phân loại dòng đọc sai thành sai dấu thanh / sai chữ cái / lặp ký tự.

Dùng: python scripts/error_analysis.py results/ft_4epochs/pred_ppocrv5_mobile_4epochs.jsonl
Căn chỉnh gt-pred theo ký tự (difflib) rồi đếm thao tác:
  tone   = thay một ký tự chỉ khác dấu thanh (sắc, huyền, hỏi, ngã, nặng)
  letter = thay/thêm/bớt một chữ cái (kể cả â/ă/ê/ô/ơ/ư/đ khác nhau)
  repeat = pred có thêm một ký tự lặp liền kề (kiểu nhưnng, nguười)
  other  = chỉ lỗi khoảng trắng / dấu câu / mất cụm cuối dòng (ngoài 3 loại của đề)
Loại của một dòng = loại có nhiều thao tác nhất (hòa: repeat > letter > tone).
Bảng 20 dòng trong docs/error_analysis_2_5.md được rà tay sau bước tự động này.
"""
import difflib, json, random, sys, unicodedata as u
from collections import Counter

TONES = {"̀", "́", "̃", "̉", "̣"}

def strip_tone(c):
    d = u.normalize("NFD", c)
    return u.normalize("NFC", "".join(x for x in d if x not in TONES))

def kind(c):
    if c.isspace(): return "space"
    if c.isalnum(): return "alnum"
    return "punct"

def ops(g, p):
    g, p = u.normalize("NFC", g), u.normalize("NFC", p)
    out = Counter()
    sm = difflib.SequenceMatcher(None, g, p, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal": continue
        gs, ps = g[i1:i2], p[j1:j2]
        if tag == "replace" and len(gs) == len(ps):
            for a, b in zip(gs, ps):
                if strip_tone(a) == strip_tone(b): out["tone"] += 1
                elif kind(a) == "alnum" and kind(b) == "alnum": out["letter"] += 1
                else: out["other"] += 1
        elif tag == "insert" or (tag == "replace" and len(ps) > len(gs)):
            for k, ch in enumerate(ps):
                prev = p[j1 + k - 1] if j1 + k > 0 else ""
                nxt = p[j1 + k + 1] if j1 + k + 1 < len(p) else ""
                if kind(ch) == "alnum" and ch.lower() in (prev.lower(), nxt.lower()): out["repeat"] += 1
                elif kind(ch) == "alnum": out["letter"] += 1
                else: out["other"] += 1
        else:  # delete / replace shorter
            if i2 == len(g) and len(gs) >= 8: out["other"] += 1      # mất cụm cuối dòng
            else:
                for ch in gs: out["letter" if kind(ch) == "alnum" else "other"] += 1
    return out

def line_class(c):
    main = {k: c[k] for k in ("repeat", "letter", "tone") if c[k]}
    if not main: return "other"
    m = max(main.values())
    for k in ("repeat", "letter", "tone"):
        if main.get(k) == m: return k

if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    wrong = []
    for r in rows:
        if u.normalize("NFC", r["gt"]) != u.normalize("NFC", r["pred"]):
            c = ops(r["gt"], r["pred"]); wrong.append((r, c, line_class(c)))
    n = len(wrong)
    cnt = Counter(k for _, _, k in wrong)
    print(f"{n}/{len(rows)} dòng sai; theo loại chính:", {k: f"{cnt[k]} ({100*cnt[k]/n:.1f}%)" for k in ("tone", "letter", "repeat", "other")})
    tot = Counter(); [tot.update(c) for _, c, _ in wrong]
    print("thao tác ký tự:", dict(tot))
    random.seed(42)
    for i, (r, c, k) in enumerate(random.sample(wrong, 20), 1):
        print(i, k, dict(c), r["image"].split("/")[-1], "\n  GT :", r["gt"], "\n  PRD:", r["pred"])
