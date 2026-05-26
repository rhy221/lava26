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

---

## Issue #7 — Mostly-Scanned PDFs Invisible to VLM (post-run discovery)

**Symptom (after first run):** Zero answers at 9.5% (59/624). Analysis of remaining zeros:
17/59 zeros came from "mostly scanned" PDFs (lc_ratio ≥ 50%) — lc=14/17 PDF → only 3 text
pages selected → 14 scanned pages completely ignored → answer never seen by model.

**Root cause:** `len(text_bearing_pages) <= max_pages` branch selects only text pages.
BM25/Dense scores scanned pages near-zero → hybrid retrieval also can't select them.
VLM receives 3 pages of text; the answer is in a scanned page → model outputs "0".

**Fix:** `run.py` + `config.yaml`
- When `lc_ratio >= cfg.retriever.mixed_scan_threshold` (default 0.4) AND MaxSim available:
  fill `max_pages - len(text_pages)` remaining slots with top scanned pages by MaxSim
- Scanned pages have `is_low_content=True` → automatically fed as VLM images

**Expected:** 17 fewer zeros → 9.5% → ~6.7% (≤ 8% target met)

---

---

## Issue #8 — `SCANNED_CHAR_THRESHOLD` Unused / `is_all_scanned` Over-triggers (LOW)

**Symptom:** `SCANNED_CHAR_THRESHOLD = 50` defined in `pdf_utils.py` but never used.
`is_all_scanned` used `num_low_content == num_pages` (< 200 chars), which could flag
image-dominant PDFs (pages with 50–199 chars of sparse text) as "all-scanned" and route
them through MaxSim-only, bypassing BM25/Dense entirely.

**Root cause:** Single `is_low_content` flag served two purposes:
1. BM25/Dense exclusion (no usable text)
2. VLM image feed trigger

These require different thresholds: a page with 150 chars of caption text is still useful
for BM25, but should still get an image fed to the VLM.

**Fix:** `src/utils/pdf_utils.py`
- `extract_pages()` now returns 5-tuple: `(page_num, text, is_low_content, is_pure_scanned, has_table)`
- `is_pure_scanned = len(text) < SCANNED_CHAR_THRESHOLD` (< 50 chars) — truly no text layer
- `is_low_content` unchanged (< 200) — still used for VLM image feed decision

**Fix:** `run.py`
- `num_pure_scanned` counter added
- `is_all_scanned = num_pure_scanned == num_pages` (stricter — only truly blank PDFs)
- `text_bearing_idx` now uses `not is_pure_scanned` — includes 50–199 char pages in BM25/Dense
- Phase 1 `is_fully_scanned` also updated to use `is_pure_scanned`
- Cache back-fill: old cache entries without `is_pure_scanned` are computed on-load

**Fix:** `_load_parsed_pages()`
- New parses write `is_pure_scanned` to the JSON cache
- On cache hit: back-fills `is_pure_scanned` from stored `text` length if field is absent

---

## Issue #9 — `has_table` Always False / Table Pages Not Sent as Images (LOW)

**Symptom:** `has_table` was hardcoded to `False` in `_load_parsed_pages()`. Pages
containing complex tables with `is_low_content=False` (≥200 chars of extracted text) were
never sent as images to the VLM, even though the table structure is often lost in text
extraction (misaligned columns, merged cells).

**Root cause:** `pdf_utils.py` had no table detection logic; `has_table: False` was a
placeholder.

**Fix:** `src/utils/pdf_utils.py`
- `extract_pages()` calls `page_obj.find_tables().tables` (PyMuPDF built-in table finder)
- `has_table = bool(found_tables)` with `try/except` fallback to `False`
- Stored in page dict and cache; used by existing `feed_image_when: [has_table]` VLM logic

**Result:** Text-rich pages that also contain tables now get images fed to the VLM, giving
the model both the extracted text and the visual table layout.

---

## Issue #10 — Case D Scanned Pages Completely Invisible (MEDIUM)

**Symptom:** Long PDFs (Case D, text_bearing > 7) may contain a few pure-scanned pages
(e.g., lc=4/70). BM25+Dense scores those scanned pages near-zero → hybrid retrieval never
selects them. If the answer is in a scanned page, the model outputs "0".

**Root cause:** The MaxSim supplement logic (Issue #7) was only in the
`elif len(text_bearing_idx) <= max_pages` branch (Case B/C). The `else` branch (Case D)
had no mechanism to rescue pure-scanned pages.

**Fix:** `run.py` — in the Case D `else` branch, after `hybrid_top_pages`:
- Compute `pure_scanned_set` = indices of pages where `is_pure_scanned=True`
- If any are unselected AND `n_room = max_pages - len(top_indices) > 0` AND MaxSim available:
  use `maxsim_score_list` to rank unselected pure-scanned pages, add top ones to fill slots
- Appends `+scan(Np)` to `retrieval_tag` when triggered

**Fix:** `src/utils/visual_retriever.py`
- Added `maxsim_score_list()` returning raw `list[float]` scores (before sorting)
- `maxsim_ranked()` refactored to call `maxsim_score_list()` — no logic duplication

---

## Issue #11 — Evidence Expansion Only Activates in Case D (LOW)

**Symptom:** `fused_scores = {}` for Cases A1, A2, B, C → `post_process.expand_evidence`
never triggered, leaving single-page predictions uncorrected for 14% + 3% + 11.5% of cases.

**Root cause:** Evidence expansion checked `if fused_scores:` which was empty for all
non-hybrid paths.

**Fix:** `run.py`
- Case A1 (all-scanned MaxSim): after computing `raw_ms = maxsim_score_list(...)`,
  populate `fused_scores = {enc_indices[i]: raw_ms[i] for i in range(...)}` so the existing
  expansion logic fires using actual MaxSim dot-product values.
- Case B (mixed): same — MaxSim scores stored in `fused_scores` after supplement run.
- Cases A2 and C: no scores available; expansion still skipped gracefully.

---

## Files Changed (Phase 2)

| File | Issues |
|---|---|
| `src/utils/pdf_utils.py` | #8, #9 |
| `src/utils/visual_retriever.py` | #10, #11 |
| `run.py` | #8, #10, #11 |

---

## Before vs After (Expected)

| Metric | Before | Target |
|---|---|---|
| Empty retrieval rate | 28.4% | < 5% |
| Vietnamese empty retrieval | 51.4% | < 10% |
| Retrieval > 7 pages | 62.2% | 0% |
| Predicted pages = 1 | 50.8% | < 35% |
| Zero answers | 13.6% | < 8% |
| Mean inference time / Q | 6.65s | < 9s |
