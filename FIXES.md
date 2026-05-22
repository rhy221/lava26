# LAVA 2026 — Bug Fixes Summary

Baseline submission score: **53%**. Target after fixes: **65–72%**.

---

## Issue #1 — Empty Retrieval Bug (CRITICAL)

**Symptom:** 177/624 questions (28.4%) had `retrieved_pages: null` or `[]`.
100% of PDFs with ≤7 pages and 0 low-content pages were affected.

**Root cause:** `src/utils/retriever.py:194` — `hybrid_top_pages` had:
```python
if n <= min_pages:   # BUG: was min_pages=3, not max_pages=7
    return list(range(n))
```
A 4-page PDF (n=4) failed the check `4 <= 3` and fell through to BM25+RRF, which could
return empty results when scores were all-zero or NaN.

**Fix:** `src/utils/retriever.py`
- Changed `n <= min_pages` → `n <= max_pages` (short-circuit for all small PDFs)
- Added fallback after `adaptive_prune`: if result is `[]`, return `list(range(min(min_pages, n)))`
- Changed return type to `tuple[list[int], dict[int, float]]` — pages + fused scores (used by Issue #4)

**Verification:** Run on q_0019 (j_0101, 4p), q_0054 (1p), q_0056 (1p), q_0603 (vi, 4p),
q_0611 (vi, 2p). `retrieved_pages` must be non-empty for all.

---

## Issue #2 — Vietnamese Tokenizer (CRITICAL)

**Symptom:** 51.4% (19/37) Vietnamese questions had empty retrieval vs 26.9% for Japanese.

**Root cause:** No dedicated Vietnamese path — Vietnamese text was handled by `_tokenize_bigram`
which splits on CJK codepoints. Vietnamese uses Latin script with tonal diacritics (ổ, ướ, …)
and multi-syllable compound words ("đầu tư", "nước ngoài") that whitespace-split alone misses.

**Fix:** `src/utils/retriever.py`
- Added `_tokenize_vietnamese()`: NFC-normalize → `underthesea.word_tokenize` if installed,
  else whitespace split + character trigrams (preserves diacritics)
- Added `"vietnamese"` dispatch in `_tokenize()`

**Fix:** `run.py`
- `tokenizer_name = "vietnamese" if language == "vi" else cfg.retriever.tokenizer`
  applied to both question and document pages (consistent tokenization)

**New dependency:** `requirements.txt` — `underthesea>=1.3.0` (optional; falls back to trigrams)

**Verification:** Tokenize "Tổng vốn đầu tư nước ngoài FDI vào Việt Nam". With underthesea:
tokens should include `["tổng vốn", "đầu tư", "nước ngoài", "fdi", "việt nam"]`.
Rerun q_0603, q_0607, q_0608, q_0610, q_0611 — ≥3/5 should have non-empty retrieval.

---

## Issue #3 — Retrieval Over-fetches 8–10 Pages (HIGH)

**Symptom:** 151 cases returned 8 pages, 237 returned 10 pages — 62% exceeded `max_pages=7`.

**Root cause:** `run.py` had an undocumented `effective_max` inflation:
```python
effective_max = (
    min(cfg.retriever.max_pages + 3, 10) if num_pages > 20   # → 10 pages
    else min(cfg.retriever.max_pages + 1, 9) if num_pages > 10  # → 8 pages
    else cfg.retriever.max_pages
)
```
This bypassed the `max_pages=7` config for ~62% of PDFs.

**Fix:** `run.py`
- Removed `effective_max` entirely; `hybrid_top_pages` now receives `cfg.retriever.max_pages` directly
- Added `assert len(top_indices) <= cfg.retriever.max_pages` to catch future regressions

**Verification:** After fix, `max(len(r["retrieved_pages"]) for r in results if r["retrieved_pages"])` ≤ 7.

---

## Issue #4 — Under-prediction of Evidence Pages (HIGH)

**Symptom:** 50.8% of predictions had exactly 1 evidence page.
Dice grounding score is capped at ≤0.67 when ground truth has 2–3 pages.

**Root cause (prompt):** The prompt had no instruction to list all contributing pages.
**Root cause (post-processing):** No mechanism to expand single-page predictions.

**Fix:** `config.yaml` — `prompts.user_template` updated with:
- Explicit instruction: "evidence_pages must contain EVERY page that contributed to the answer"
- Contrasting single-page vs multi-page examples
- Instruction to include all source pages for comparisons/sums

**Fix:** `run.py` — post-processing after `parse_model_output`:
- If `len(pred_pages) == 1` AND `len(selected_nums) >= 3` AND `fused_scores` is available,
  sort retrieved pages by RRF score, check if top-2 gap ≤ `expand_evidence_score_diff` (10%),
  and if so, add the top-2 page to `pred_pages`
- Toggled by `config.yaml: post_process.expand_evidence: true`

**New config parameters:**
```yaml
post_process:
  expand_evidence: true
  expand_evidence_score_diff: 0.10
```

**Note:** `fused_scores` is only populated for the hybrid BM25+Dense path (large PDFs).
For small PDFs (all-text path) and all-scanned PDFs, expansion is skipped gracefully.

**Verification:** Mean predicted_pages count should rise from ~1.5 toward 1.8–2.2.
Single-page predictions should drop below 35%.

---

## Issue #5 — Zero Answers Too Readily Given (MEDIUM)

**Symptom:** 85 questions (13.6%) returned `"0"` or `["0"]`, independent of retrieval size.

**Root cause:** Prompt had a permissive escape hatch:
> "If the answer cannot be found in the document: use '0' or ['0']"
The model used this as a low-effort fallback even when relevant content existed.

**Fix:** `config.yaml` — `prompts.user_template` updated:
- `"0"` is now reserved ONLY for when "the document contains absolutely no relevant information"
- Explicit instruction: "For uncertain cases, provide your best inference from the closest evidence"
- Added chain-of-thought scaffold: "First, identify in one sentence which retrieved page is most relevant"

**Verification:** Rerun the 85 currently-zero questions. Expected ≥30–40% should now return
non-zero. Non-zero accuracy on 100 random cases should not regress.

---

## Issue #6 — All-Scanned PDFs Poorly Handled (LOW)

**Symptom:** 114/122 mostly-scanned PDFs had empty `retrieved_pages` despite MaxSim support.
VLM received images without knowing which page each image corresponded to.

**Root cause 1:** For all-scanned PDFs, `top_indices = list(range(num_pages))` (all pages),
so `is_all_pages = True` and `retrieved_pages = null` in results.json — grounding was blind.

**Root cause 2:** MaxSim was run separately for IMAGE selection but NOT for `retrieved_pages`.
The VLM received images with no labeling of which image = which page, making evidence_pages
attribution impossible.

**Root cause 3:** MaxSim was run twice (once in retrieval section, once in VLM section).

**Fix:** `run.py`
- All-scanned PDFs now use MaxSim in the retrieval section to determine `top_indices`
  (top `max_pages` by ColQwen MaxSim), so `retrieved_pages` is non-null in results.json
- VLM images section reuses the same `top_indices` (no duplicate MaxSim run)
- Added image-page label prepended to `document_text`:
  `"[Scanned document. Image-to-page mapping: Image 1=Page 3, Image 2=Page 7, ...]"`
  so the model knows which image corresponds to which page number

**Verification:** Rerun 10 all-scanned PDFs. Expected: `retrieved_pages` non-empty,
`predicted_pages` subset of `retrieved_pages` for most cases.

---

## Files Changed

| File | Issues |
|---|---|
| `src/utils/retriever.py` | #1, #2 |
| `run.py` | #2, #3, #4, #6 |
| `config.yaml` | #4, #5 + `post_process` section |
| `requirements.txt` | #2 (underthesea) |
| `scripts/verify_fixes.py` | New — verification script |

## Before vs After (Expected)

| Metric | Before | Target |
|---|---|---|
| Empty retrieval rate | 28.4% | < 5% |
| Vietnamese empty retrieval | 51.4% | < 10% |
| Retrieval > 7 pages | 62.2% | 0% |
| Predicted pages = 1 | 50.8% | < 35% |
| Zero answers | 13.6% | < 8% |
| Mean inference time / Q | 6.65s | < 9s |
