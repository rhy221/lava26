"""Analyze inference case distribution from results.json."""
import json
from collections import defaultdict

with open("results.json", encoding="utf-8") as f:
    results = json.load(f)

MAX_PAGES = 7


def classify(r):
    n = r.get("num_pages", 0) or 1
    lc = r.get("num_low_content", 0)
    ret = r.get("retrieved_pages")
    is_all_scanned = lc == n
    text_bearing = n - lc
    lc_ratio = lc / n
    if is_all_scanned:
        return "A2" if ret is None else "A1"
    if text_bearing <= MAX_PAGES:
        return "B" if lc_ratio >= 0.4 else "C"
    return "D"


cases = defaultdict(list)
for r in results:
    cases[classify(r)].append(r)

total = len(results)
print(f"Total questions: {total}")

LABELS = {
    "A1": "All-scanned, MaxSim used        (ret != null)",
    "A2": "All-scanned, MaxSim unavailable  (ret == null)",
    "B":  "Mostly-scanned mixed PDF         (lc_ratio >= 0.4, text_bearing <= 7)",
    "C":  "Short text-only PDF              (text_bearing <= 7, lc_ratio < 0.4)",
    "D":  "Long PDF, hybrid BM25+Dense      (text_bearing > 7)",
}

for case_id in ["A1", "A2", "B", "C", "D"]:
    rs = cases[case_id]
    n = len(rs)
    pct = 100 * n / total

    zeros = sum(
        1 for r in rs
        if str(r.get("predicted_answer", "")).strip() in ("0", '["0"]', "['0']")
    )
    zero_pct = 100 * zeros / max(n, 1)

    times = [r["time_seconds"] for r in rs if "time_seconds" in r]
    avg_t = sum(times) / len(times) if times else 0.0

    ret_lens = [len(r["retrieved_pages"]) for r in rs if isinstance(r.get("retrieved_pages"), list)]
    avg_ret = sum(ret_lens) / len(ret_lens) if ret_lens else 0.0

    langs = defaultdict(int)
    for r in rs:
        langs[r.get("language", "?")] += 1
    lang_str = "  ".join(f"{k}:{v}" for k, v in sorted(langs.items()))

    fmts = defaultdict(int)
    for r in rs:
        fmts[r.get("answer_format", "?")] += 1
    fmt_str = "  ".join(f"{k}:{v}" for k, v in sorted(fmts.items()))

    pages_list = [r.get("num_pages", 0) for r in rs]
    if pages_list:
        pg_str = f"{min(pages_list)}-{max(pages_list)}, avg {sum(pages_list)/len(pages_list):.1f}"
    else:
        pg_str = "N/A"

    pred_lens = [len(r["predicted_pages"]) for r in rs if isinstance(r.get("predicted_pages"), list)]
    avg_pred = sum(pred_lens) / len(pred_lens) if pred_lens else 0.0
    single = sum(1 for x in pred_lens if x == 1)
    single_pct = 100 * single / max(len(pred_lens), 1)

    print()
    print(f"[{case_id}] {LABELS[case_id]}")
    print(f"  Count:           {n:4d}  ({pct:.1f}%)")
    print(f"  Zero answers:    {zeros:4d}  ({zero_pct:.1f}%)")
    print(f"  Avg time:        {avg_t:.2f}s")
    print(f"  Avg ret pages:   {avg_ret:.2f}")
    print(f"  Avg pred pages:  {avg_pred:.2f}  (single-page: {single_pct:.1f}%)")
    print(f"  Pages range:     {pg_str}")
    print(f"  Languages:       {lang_str}")
    print(f"  Formats:         {fmt_str}")
