"""
Verification script for LAVA 2026 bug fixes.

Loads results.json from a run and reports the before/after metrics
described in FIXES.md. Pass --results <path> to compare a new run.
Pass --baseline <path> to compare two runs side-by-side.

Usage:
    python scripts/verify_fixes.py --results runs/after/results.json
    python scripts/verify_fixes.py --results runs/after/results.json \\
                                   --baseline runs/before/results.json
"""

import argparse
import json
from pathlib import Path


def compute_metrics(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    # Empty retrieval: null means "all pages used" (not empty), [] is truly empty
    empty_retrieval = sum(1 for r in results if r.get("retrieved_pages") == [])
    null_retrieval  = sum(1 for r in results if r.get("retrieved_pages") is None)

    # Over-fetch: retrieved_pages longer than 7
    over_fetch = sum(
        1 for r in results
        if isinstance(r.get("retrieved_pages"), list) and len(r["retrieved_pages"]) > 7
    )

    # Vietnamese subset
    vi_results = [r for r in results if r.get("language") == "vi"]
    vi_empty = sum(1 for r in vi_results if r.get("retrieved_pages") == [])

    # Single-page evidence
    single_page = sum(
        1 for r in results if isinstance(r.get("predicted_pages"), list)
        and len(r["predicted_pages"]) == 1
    )
    zero_answers = sum(
        1 for r in results
        if str(r.get("predicted_answer", "")) in ("0", "['0']", '["0"]')
    )
    mean_predicted_pages = (
        sum(len(r["predicted_pages"]) for r in results
            if isinstance(r.get("predicted_pages"), list))
        / max(1, sum(1 for r in results if isinstance(r.get("predicted_pages"), list)))
    )
    times = [r["time_seconds"] for r in results if "time_seconds" in r]
    mean_time = sum(times) / len(times) if times else 0

    # Score metrics (train split only)
    has_scores = any("vqa_score" in r for r in results)
    mean_vqa = mean_grounding = mean_overall = None
    if has_scores:
        scored = [r for r in results if "vqa_score" in r]
        mean_vqa       = sum(r["vqa_score"] for r in scored) / len(scored)
        mean_grounding = sum(r["grounding_score"] for r in scored) / len(scored)
        mean_overall   = sum(r["overall_score"] for r in scored) / len(scored)

    return {
        "n": n,
        "empty_retrieval_pct":     round(100 * empty_retrieval / n, 1),
        "null_retrieval_pct":      round(100 * null_retrieval / n, 1),
        "over_fetch_pct":          round(100 * over_fetch / n, 1),
        "vi_n":                    len(vi_results),
        "vi_empty_pct":            round(100 * vi_empty / max(1, len(vi_results)), 1),
        "single_page_evidence_pct": round(100 * single_page / n, 1),
        "zero_answers_pct":        round(100 * zero_answers / n, 1),
        "mean_predicted_pages":    round(mean_predicted_pages, 2),
        "mean_time_sec":           round(mean_time, 2),
        "mean_vqa":                round(mean_vqa, 4) if mean_vqa is not None else None,
        "mean_grounding":          round(mean_grounding, 4) if mean_grounding is not None else None,
        "mean_overall":            round(mean_overall, 4) if mean_overall is not None else None,
    }


TARGETS = {
    "empty_retrieval_pct":      (None, 5.0,  "< 5%"),
    "over_fetch_pct":           (None, 0.0,  "= 0%"),
    "vi_empty_pct":             (None, 10.0, "< 10%"),
    "single_page_evidence_pct": (None, 35.0, "< 35%"),
    "zero_answers_pct":         (None, 8.0,  "< 8%"),
    "mean_time_sec":            (None, 9.0,  "< 9s"),
}


def print_report(label: str, m: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  (n={m['n']})")
    print(f"{'='*60}")

    rows = [
        ("Empty retrieval ([])",         f"{m['empty_retrieval_pct']}%",  "empty_retrieval_pct"),
        ("Null retrieval (all pages)",   f"{m['null_retrieval_pct']}%",   None),
        ("Retrieval > 7 pages",          f"{m['over_fetch_pct']}%",       "over_fetch_pct"),
        (f"Vietnamese empty ({m['vi_n']} q)", f"{m['vi_empty_pct']}%",   "vi_empty_pct"),
        ("Predicted pages = 1",          f"{m['single_page_evidence_pct']}%", "single_page_evidence_pct"),
        ("Zero answers",                 f"{m['zero_answers_pct']}%",     "zero_answers_pct"),
        ("Mean predicted pages",         str(m["mean_predicted_pages"]),  None),
        ("Mean time / question",         f"{m['mean_time_sec']}s",        "mean_time_sec"),
    ]
    if m.get("mean_vqa") is not None:
        rows += [
            ("Mean VQA score",   str(m["mean_vqa"]),       None),
            ("Mean Grounding",   str(m["mean_grounding"]), None),
            ("Mean Overall",     str(m["mean_overall"]),   None),
        ]

    for name, value, key in rows:
        if key and key in TARGETS:
            _lo, hi, desc = TARGETS[key]
            val_f = float(value.rstrip("%s"))
            ok = val_f <= hi if hi is not None else True
            status = "PASS" if ok else "FAIL"
            print(f"  {name:<35} {value:<10} [{status}] target {desc}")
        else:
            print(f"  {name:<35} {value}")


def compare(before: dict, after: dict) -> None:
    print(f"\n{'='*60}")
    print("  COMPARISON  (before → after)")
    print(f"{'='*60}")

    metrics = [
        ("Empty retrieval",          "empty_retrieval_pct",      "< 5%"),
        ("Retrieval > 7 pages",      "over_fetch_pct",           "= 0%"),
        ("Vietnamese empty",         "vi_empty_pct",             "< 10%"),
        ("Predicted pages = 1",      "single_page_evidence_pct", "< 35%"),
        ("Zero answers",             "zero_answers_pct",         "< 8%"),
        ("Mean time / question",     "mean_time_sec",            "< 9s"),
        ("Mean predicted pages",     "mean_predicted_pages",     "↑"),
    ]
    score_metrics = [
        ("Mean VQA",     "mean_vqa"),
        ("Mean Ground",  "mean_grounding"),
        ("Mean Overall", "mean_overall"),
    ]

    for name, key, target in metrics:
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            continue
        arrow = "→"
        change = f"{b} {arrow} {a}"
        ok = "PASS" if a <= float(target.strip("<>=%s").strip()) else "FAIL" if "%" in target or "s" in target else ""
        print(f"  {name:<30} {change:<20} target {target}  {ok}")

    for name, key in score_metrics:
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            continue
        delta = f"{a - b:+.4f}" if a is not None and b is not None else ""
        print(f"  {name:<30} {b} → {a}  ({delta})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",  required=True, help="Path to after-fix results.json")
    parser.add_argument("--baseline", default=None,  help="Path to before-fix results.json")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        after_results = json.load(f)

    after_metrics = compute_metrics(after_results)
    print_report(f"AFTER — {args.results}", after_metrics)

    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            before_results = json.load(f)
        before_metrics = compute_metrics(before_results)
        print_report(f"BEFORE — {args.baseline}", before_metrics)
        compare(before_metrics, after_metrics)

    print()


if __name__ == "__main__":
    main()
