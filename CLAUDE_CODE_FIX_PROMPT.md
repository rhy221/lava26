# LAVA 2026 Pipeline Audit & Fix Prompt

You are auditing and fixing the LAVA 2026 Document VQA pipeline. The current submission scores **53%** (target: 65-72% after fixes). A detailed analysis of `results.json` (624 questions) has identified **6 concrete bugs** with strong statistical evidence. Your job is to **locate each bug in the source code, fix it correctly, and verify the fix**.

---

## Project context

- **Entry point:** `run.py` (~839 lines)
- **Config:** `config.yaml` (OmegaConf, source of truth)
- **Key modules:**
  - `src/utils/retriever.py` — BM25, E5 dense, RRF fusion, adaptive pruning
  - `src/utils/visual_retriever.py` — ColQwen, MaxSim
  - `src/utils/pdf_utils.py` — PyMuPDF extraction, token budgeting
  - `src/utils/output_parser.py` — JSON parsing, CSV formatting
  - `src/evaluate.py` — scoring metrics
- **Models:** Qwen3.6-27B (VLM), E5-large-multilingual (dense), ColQwen2.5 (visual)

---

## Required working method

1. **Investigate before changing anything.** For each issue, first read the relevant code, form a hypothesis about the root cause, and verify the hypothesis matches the symptom data below. Do not guess.
2. **Make one fix at a time.** After each fix, run a small verification (described per issue) before moving on.
3. **Preserve existing behavior** for cases that already work. The 8.8% of cases retrieving 2-7 pages correctly should still retrieve 2-7 pages.
4. **Use the config when introducing parameters.** Do not hardcode magic numbers; add them under the appropriate config section.
5. **Log your reasoning.** Before each code change, write a comment block explaining: (a) what bug you found, (b) why this code caused it, (c) what your fix does.

---

## Issue #1 (CRITICAL — fix first): Empty retrieval bug

### Symptom data
- **177 out of 624 questions (28.4%)** have `retrieved_pages: []` or `null`.
- **100% of PDFs with ≤7 pages and 0 low-content pages** return empty retrieval — this is supposed to be the easiest case.
- Concrete failing examples: `q_0019` (4 pages, 0 scanned), `q_0054` (1 page), `q_0056` (1 page), `q_0603` (4 pages, Vietnamese).
- Pipeline documentation states: *"Tổng trang ≤ max_pages → dùng tất cả trang có text"* — but code does not implement this.

### Investigation steps
1. Open `src/utils/retriever.py`. Locate the function that returns the final list of retrieved pages (likely named `retrieve()`, `select_pages()`, or similar).
2. Trace the path for a hypothetical PDF with `num_pages=4`. Where does the code decide what to return?
3. Check the adaptive pruning logic (`threshold = 0.8 × max_score`). Could it be filtering out everything when scores are very low or NaN?
4. Check tokenization for short documents — does BM25 fail when the document has very few tokens?
5. Check the dense path — is the cache file `cache/dense/{file_id}.npy` actually being created for small PDFs?

### Hypotheses to verify (in order of likelihood)
- **H1:** The "use all pages" fallback for small PDFs is missing entirely from the code.
- **H2:** Adaptive pruning with `0.8 × max_score` removes everything when max_score is below some implicit floor.
- **H3:** A try/except swallows an exception silently, returning `[]`.
- **H4:** The function returns early when `len(pages) <= max_pages` thinking "no need to retrieve" but forgets to return the page list.

### Fix requirements
- If the PDF has ≤ `max_pages` pages with extractable text, return **all of those pages** (sorted ascending).
- If retrieval scores are all zero or NaN, fall back to returning the first `min_pages` pages (not an empty list).
- The fix must be at the retriever level, not patched at the caller level — other call sites should benefit too.

### Verification
After fixing, run inference on these 5 question IDs and confirm `retrieved_pages` is non-empty:
- `q_0019` (file `j_0101`, 4 pages)
- `q_0054` (file unknown, 1 page)
- `q_0056` (1 page)
- `q_0603` (Vietnamese, 4 pages)
- `q_0611` (Vietnamese, 2 pages)

---

## Issue #2 (CRITICAL): Vietnamese tokenizer broken

### Symptom data
- **51.4% (19/37)** of Vietnamese questions have empty retrieval, vs. 26.9% for Japanese.
- Pipeline doc says tokenizer "Bigram (default): Vietnamese, Latin — CJK character bigrams + whitespace split for Latin/Vietnamese" — but Vietnamese is being treated as CJK-style, which is wrong.
- Vietnamese uses Latin script with diacritics and multi-syllable words separated by spaces.

### Investigation steps
1. Open `src/utils/retriever.py`. Find the tokenizer selection logic — likely a function `tokenize()` or branching by `language` field.
2. Trace what happens when `language == "vi"`. Does it go through the Japanese MeCab path? The CJK-bigram path? Whitespace split?
3. Test on the sample Vietnamese query: `"Tổng vốn đầu tư nước ngoài (FDI) vào Việt Nam trong 2 tháng đầu năm 2026 đạt bao nhiêu tỷ USD?"` — print the resulting tokens. Are they useful for BM25 (multi-character word tokens) or garbage (single chars, broken bigrams)?

### Fix requirements
- Add a dedicated Vietnamese tokenizer path. Preferred order:
  1. **Option A (best quality):** Use `underthesea.word_tokenize()` if installable. Add `underthesea` to `requirements.txt`.
  2. **Option B (fallback):** Use lowercase + Unicode-aware whitespace split + character trigrams as a hybrid. Vietnamese diacritics must be preserved (NFC normalization).
- Apply tokenization consistently to **both** the question and the document pages — mismatch = no recall.
- Lowercase normalization should be applied to both sides.

### Verification
- Tokenize the test query above and confirm tokens look like `["tổng", "vốn", "đầu tư", "nước ngoài", "fdi", "việt nam", ...]` (or trigrams that meaningfully cover keywords).
- Rerun inference on `q_0603`, `q_0607`, `q_0608`, `q_0610`, `q_0611` — at least 3 of 5 should now have non-empty retrieval.

---

## Issue #3 (HIGH): Retrieval over-fetches 8-10 pages despite max_pages=7

### Symptom data
- `config.yaml` says `retriever.max_pages: 7`.
- Actual distribution: **151 cases return 8 pages, 237 cases return 10 pages** (62% of all cases ignore the cap).
- These over-fetched cases are predominantly **text PDFs**, not scanned, so this is not the visual fallback path.

### Investigation steps
1. Find every place that builds the final retrieved_pages list in `src/utils/retriever.py` (and possibly `run.py`).
2. Search for the literals `7`, `8`, `10`, and `max_pages` across the codebase. Look for places where pages are appended after the cap.
3. Hypothesis: there are **multiple retrieval lists** (BM25 top-K, Dense top-K, possibly ColQwen) being **unioned** rather than RRF-fused, then the cap is forgotten.
4. Hypothesis: top-K for individual retrievers is set to a higher number (e.g., 10) for diversity, then RRF combines them but the final cap is not applied.

### Fix requirements
- After RRF fusion and adaptive pruning, **enforce a hard cap**: `final_pages = sorted_pages[:max_pages]`.
- Add an `assert len(final_pages) <= cfg.retriever.max_pages` to catch regressions.
- If different `top_k` values are used per retriever internally, that is fine, but the merged output must respect `max_pages`.

### Verification
- After fix, distribution of `len(retrieved_pages)` across all 624 questions must satisfy: `max(counts) ≤ 7`.
- The 8.8% of cases that already returned 2-7 pages should be unchanged.

---

## Issue #4 (HIGH): Under-prediction of evidence_pages (Dice score capped)

### Symptom data
- **50.8% (317/624)** of predictions have exactly **1 evidence page**.
- 35.4% have 2 pages, 7.2% have 0 pages.
- Grounding metric is Dice: `2 × |A∩B| / (|A| + |B|)`. When ground truth has 2-3 pages and prediction has 1, Dice is capped at 0.5-0.67 even if that 1 page is correct.

### Investigation steps
1. Open `run.py` and find the prompt template sent to the VLM. Look for the format rules section.
2. The current prompt likely says something generic like `"evidence_pages": [1]` without instructing the model to list **all** pages used.
3. Open `src/utils/output_parser.py`. Verify it accepts arrays of any length (not silently truncating).

### Fix requirements
Modify the prompt to **explicitly instruct multi-page evidence reporting**. The new prompt section for evidence rules must:
- State that evidence_pages must contain **every page** that contributed to the answer.
- Give two contrasting examples: one with 1 page (when answer is fully on one page), one with multiple pages (e.g., when comparing values across pages, or when answer combines info from a header on page X and a table on page Y).
- For `number` format questions involving comparisons/sums, instruct the model to include all source pages of the numbers.

Additionally, add a **post-processing safety net** in `run.py` or `output_parser.py`:
- If `predicted_pages` has exactly 1 element AND retrieval found ≥3 pages with closely-clustered scores (top-1 score vs top-2 score within 10%), expand `predicted_pages` to include the top-2 retrieved page as well. Make this behavior toggleable via `config.yaml: post_process.expand_evidence: true`.

### Verification
- After fix, rerun on a sample of 50 questions with `answer_format in ["unordered_list", "number"]`.
- Expected: mean predicted_pages count rises from 1.5 toward **1.8-2.2**.
- Single-page predictions should drop from 50.8% to below 35%.

---

## Issue #5 (MEDIUM): Zero answers too readily given (13.6% of all cases)

### Symptom data
- 85 questions (13.6%) returned `"0"` or `["0"]` as the answer.
- This rate is **independent of retrieval size** (10-14% for retrieval=7, 8, 10), meaning it's not a context-availability problem — the model is **giving up** too easily.
- Current prompt explicitly tells the model: *"Nếu không tìm thấy: answer = '0' hoặc ['0']"* — this is the escape hatch.

### Investigation steps
1. Locate the prompt template (same place as Issue #4).
2. Identify the exact escape-hatch instruction.

### Fix requirements
Replace the escape-hatch instruction with a more nuanced one:
- Reserve `"0"` ONLY for cases where the document explicitly states a quantity of zero, OR the question demonstrably has no answerable content in the retrieved pages.
- For uncertain cases, instruct the model to provide its **best inference** from the closest evidence, not to default to `"0"`.
- Add a brief chain-of-thought scaffold: ask the model to first state which retrieved page is most relevant in one sentence, then provide the JSON answer.

### Verification
- Rerun inference on the 85 currently-zero questions.
- Expected: at least 30-40% of these should now return non-zero answers (some will still be genuinely unanswerable).
- Watch for regression: the answer accuracy on **non-zero** cases should not drop. Run on 100 random non-zero cases to confirm.

---

## Issue #6 (LOW): All-scanned PDFs poorly handled

### Symptom data
- 122 PDFs are mostly scanned (>80% low-content pages).
- 114 of these have empty `retrieved_pages` despite the pipeline doc claiming ColQwen MaxSim handles this case.
- Example: `q_0050` has PDF with 99 pages, retrieved 9 pages via what looks like visual retrieval, but model output `evidence_pages: []`.

### Investigation steps
1. In `src/utils/visual_retriever.py`, find the path used for all-scanned PDFs.
2. Verify that `cache/visual/{file_id}.pt` and `cache/query_embs.pt` are actually created during Phase 1 for these PDFs.
3. Trace whether the MaxSim-selected pages are passed through to `retrieved_pages` (the field saved to `results.json`) or only used internally for image selection.

### Fix requirements
- Ensure ColQwen-selected pages are recorded in `retrieved_pages` for all-scanned PDFs (so they appear in the output JSON for grounding).
- Ensure the VLM prompt for all-scanned PDFs explicitly states which page each image corresponds to, e.g., `"Image 1 = Page 3, Image 2 = Page 7, ..."`. Otherwise the model cannot ground evidence_pages correctly.

### Verification
- Rerun on 10 all-scanned PDFs.
- Expected: `retrieved_pages` is non-empty, and `predicted_pages` is a subset of `retrieved_pages` for most cases.

---

## Final integration check

After all 6 fixes, run a full inference pass on a 50-question subset stratified by:
- Language: 25 Japanese + 25 Vietnamese
- Answer format: mix of string, number, unordered_list
- PDF size: mix of small (≤7 pages), medium (8-30), large (>30)
- PDF type: mix of text-heavy and scan-heavy

Report the following statistics, comparing **before fix** vs **after fix**:

| Metric | Before (baseline) | After (target) |
|---|---|---|
| Empty retrieval rate | 28.4% | < 5% |
| Vietnamese empty retrieval | 51.4% | < 10% |
| Retrieval > 7 pages | 62.2% | 0% |
| Predicted pages = 1 | 50.8% | < 35% |
| Zero answers | 13.6% | < 8% |
| Mean inference time per Q | 6.65s | < 9s (slight increase OK) |

If any metric regresses, investigate before submitting.

---

## What NOT to change

- **Do not change the model** (Qwen3.6-27B stays).
- **Do not change RRF fusion to weighted sum** — RRF is the right choice and the doc explains why.
- **Do not remove the visual retrieval pathway** for all-scanned PDFs — it just needs to actually feed pages to the VLM.
- **Do not refactor the two-phase architecture** — Phase 1 cache + Phase 2 inference is sound.
- **Do not change quantization settings** (bnb 4-bit NF4 stays) — it fits the 40GB A100 budget.

---

## Deliverables

1. A `FIXES.md` file at repo root summarizing each fix: what bug was found, exact file and function changed, before/after code snippets, verification result.
2. Updated source files with inline comments explaining each fix.
3. Updated `config.yaml` if new tunable parameters were added.
4. Updated `requirements.txt` if new dependencies (e.g., `underthesea`) were added.
5. A small verification script `scripts/verify_fixes.py` that runs the 50-question stratified check and prints the before/after table.

Begin with Issue #1. Do not move to the next issue until the current one is verified.
