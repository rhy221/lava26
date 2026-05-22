"""
Evaluation metrics for LAVA 2026.

VQA scoring uses string normalization (fast proxy).
The real competition judge uses Gemma-3 1B — integrate later if needed.
"""
import ast
import re
import unicodedata


# ── VQA Score ────────────────────────────────────────────────────────────────

def vqa_score(predicted: str, ground_truth: str, answer_format: str) -> float:
    if answer_format in ("string", "number"):
        return _string_score(predicted, ground_truth)
    if answer_format == "unordered_list":
        return _f1_score(_parse_list(predicted), _parse_list(ground_truth))
    if answer_format == "ordered_list":
        return _lcs_score(_parse_list(predicted), _parse_list(ground_truth))
    return 0.0


def _normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(",", "")
    return text


def _string_score(pred: str, gt: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gt) else 0.0


def _parse_list(list_str: str) -> list[str]:
    try:
        parsed = ast.literal_eval(str(list_str).strip())
        if isinstance(parsed, list):
            return [_normalize(str(x)) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return [_normalize(str(list_str).strip())]


def _f1_score(pred_items: list[str], gt_items: list[str]) -> float:
    if not gt_items:
        return 1.0 if not pred_items else 0.0
    if not pred_items:
        return 0.0
    gt_remaining = list(gt_items)
    matches = 0
    for p in pred_items:
        if p in gt_remaining:
            matches += 1
            gt_remaining.remove(p)
    precision = matches / len(pred_items)
    recall = matches / len(gt_items)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_score(pred_items: list[str], gt_items: list[str]) -> float:
    if not gt_items and not pred_items:
        return 1.0
    if not gt_items or not pred_items:
        return 0.0
    n, m = len(pred_items), len(gt_items)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_items[i - 1] == gt_items[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m] / max(n, m)


# ── Grounding Score ──────────────────────────────────────────────────────────

def grounding_score(predicted_pages: list[int], ground_truth_pages: list[int]) -> float:
    """Dice coefficient: 2 * |pred ∩ gt| / (|pred| + |gt|)"""
    if not ground_truth_pages and not predicted_pages:
        return 1.0
    if not ground_truth_pages or not predicted_pages:
        return 0.0
    pred_set = set(predicted_pages)
    gt_set = set(ground_truth_pages)
    intersection = len(pred_set & gt_set)
    return 2 * intersection / (len(pred_set) + len(gt_set))


# ── Overall ──────────────────────────────────────────────────────────────────

def overall_score(vqa: float, grounding: float) -> float:
    return (vqa + grounding) / 2


# ── Aggregate ────────────────────────────────────────────────────────────────

def aggregate_results(results: list[dict]) -> dict:
    if not results:
        return {}
    mean_vqa = sum(r["vqa_score"] for r in results) / len(results)
    mean_grounding = sum(r["grounding_score"] for r in results) / len(results)
    mean_overall = sum(r["overall_score"] for r in results) / len(results)
    mean_time = sum(r.get("time_seconds", 0) for r in results) / len(results)
    total_time = sum(r.get("time_seconds", 0) for r in results)
    return {
        "num_questions": len(results),
        "mean_vqa_score": round(mean_vqa, 4),
        "mean_grounding_score": round(mean_grounding, 4),
        "mean_overall_score": round(mean_overall, 4),
        "avg_time_per_question_sec": round(mean_time, 2),
        "total_time_sec": round(total_time, 2),
        "estimated_624q_hours": round(mean_time * 624 / 3600, 2),
    }
