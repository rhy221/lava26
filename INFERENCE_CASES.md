# LAVA 2026 — Phân Tích Chi Tiết Input Inference

> Tài liệu này mô tả toàn bộ các trường hợp có thể xảy ra khi pipeline chuẩn bị input
> cho VLM (Qwen3.6-27B), sau tất cả các bản vá từ FIXES.md.
> Code tham chiếu: `run.py` lines 547–748.

---

## Tổng Quan: 2 Giai Đoạn Độc Lập

```
PDF + Question
      │
      ▼
┌─────────────────────────────────────────┐
│  GIAI ĐOẠN 1 — RETRIEVAL               │
│  Quyết định: trang nào đưa vào context │
│  Output: top_indices, document_text    │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  GIAI ĐOẠN 2 — VLM IMAGES              │
│  Quyết định: ảnh nào đính kèm prompt   │
│  Output: page_images_for_vlm (0–6 img) │
└────────────────────┬────────────────────┘
                     │
                     ▼
         Input cuối cho Qwen VLM:
         [system_prompt]
         [document_text + ảnh]
         [question + format rules]
```

Hai giai đoạn **độc lập**: retrieval xác định ngữ cảnh text, image selection xác định
ngữ cảnh hình ảnh. Kết quả của giai đoạn 1 ảnh hưởng đến giai đoạn 2 (ảnh được chọn
từ tập trang đã retrieve), nhưng logic quyết định khác nhau.

---

## Giai Đoạn 1 — Retrieval

### Biến Phân Loại

| Biến | Tính từ | Ý nghĩa |
|---|---|---|
| `is_all_scanned` | `num_low_content == num_pages` | 100% trang không có text layer |
| `text_bearing_idx` | `[i for i if not is_low_content]` | Danh sách index trang có text |
| `lc_ratio` | `num_low_content / num_pages` | Tỉ lệ trang scan |
| `doc_visual_emb` | load từ `cache/visual/{file_id}.pt` | ColQwen embeddings (có hoặc không) |
| `n_slots` | `max_pages - len(text_bearing_idx)` | Slot còn trống để bổ sung |

### Decision Tree

```
                        is_all_scanned?
                       /               \
                     YES                NO
                      │                 │
          doc_visual_emb            len(text_bearing_idx)
           AND q_emb có?             <= max_pages (7)?
           /         \               /              \
         YES          NO           YES               NO
          │            │            │                 │
        [A1]         [A2]    lc_ratio >= 0.4?      [D]
                              /          \         BM25+
                            YES           NO       Dense
                    doc_visual_emb?               Hybrid
                    /           \
                  YES            NO
                   │              │
                 [B]             [C]
```

---

### Case A1 — Toàn Scan + MaxSim Có

**Điều kiện:**
- `is_all_scanned = True` (lc_ratio = 100%)
- `doc_visual_emb is not None` (cache ColQwen tồn tại)
- `precomp_query_embs[q_id]` tồn tại

**Cơ chế:**
```
query_emb (cache Phase 1)
    + doc_emb của tất cả trang (ColQwen embeddings)
    → MaxSim scoring: score(page) = max over query_tokens [ max(q · p) ]
    → Rank tất cả trang giảm dần
    → Lấy top max_pages (7) trang
    → Sort theo thứ tự trang (ascending)
```

**Output:**
- `top_indices` = 7 trang liên quan nhất (theo MaxSim)
- `retrieval_tag` = `maxsim_visual(7p)`
- `document_text` = markers rỗng (`=== Page N ===\n\n`) vì không có text
- `selected_nums` ≠ all pages → `retrieved_pages` = danh sách cụ thể trong results.json

**Ví dụ:** PDF j_0117 (30 trang, lc=30/30) → MaxSim chọn [8,12,13,16,18,20,21]

---

### Case A2 — Toàn Scan + MaxSim Không Có

**Điều kiện:**
- `is_all_scanned = True`
- `doc_visual_emb is None` HOẶC `q_emb` không có trong cache

**Cơ chế:**
```
Fallback đơn giản: lấy min(max_pages, num_pages) trang đầu tiên
top_indices = [0, 1, 2, ..., min(6, N-1)]
```

**Output:**
- `top_indices` = 7 trang đầu
- `retrieval_tag` = `seq_fallback(7p)`
- `document_text` = markers rỗng
- **Chất lượng thấp** — không biết trang nào chứa câu trả lời

**Nguyên nhân xảy ra:** Phase 1 chưa chạy, hoặc ColQwen load lỗi, hoặc PDF mới thêm
vào sau khi đã preprocess.

---

### Case B — Mostly Scan + MaxSim Bổ Sung

**Điều kiện:**
- `is_all_scanned = False` (có ít nhất 1 trang text)
- `len(text_bearing_idx) <= max_pages` (ít trang text, ≤7)
- `lc_ratio >= mixed_scan_threshold` (mặc định 0.4)
- `doc_visual_emb is not None` AND `q_emb` có trong cache
- `n_slots = max_pages - len(text_bearing_idx) > 0`

**Cơ chế:**
```
Bước 1: Lấy tất cả trang text (text_bearing_idx) → thường 2–5 trang
Bước 2: Tính n_slots = max_pages - len(text_pages)
Bước 3: MaxSim rank tất cả trang
Bước 4: Lọc MaxSim kết quả → chỉ lấy trang có is_low_content=True
         và chưa có trong top_indices
Bước 5: Lấy n_slots trang scanned đầu tiên từ MaxSim ranking
Bước 6: Merge = text_pages ∪ top_scanned_pages → sort theo thứ tự trang
```

**Output:**
- `top_indices` = text pages + MaxSim scanned pages (tổng ≤ 7)
- `retrieval_tag` = `mixed_text+maxsim(Np)`
- `document_text` = text thật của text pages + markers rỗng của scanned pages
- Scanned pages có `is_low_content=True` → **sẽ được gửi kèm ảnh** ở Giai Đoạn 2

**Ví dụ:** PDF lc=14/17 (3 text pages + 14 scanned)
```
text_bearing_idx = [8, 13, 15]   → 3 trang text
n_slots = 7 - 3 = 4
MaxSim rank scanned pages → [3, 6, 10, 11, 2, 5, ...]
extra = [3, 6, 10, 11]          → 4 trang scanned thêm
top_indices = [3, 6, 8, 10, 11, 13, 15]  → 7 trang tổng
```

**Tại sao cần fix này:** Trước đây chỉ chọn 3 text pages. Model nhận 3 trang text rỗng,
câu trả lời nằm trong 14 trang scan → model output "0". Fix này đã giải quyết 17/59 zero
answers sau lần chạy đầu.

---

### Case C — Text Ít, Lc Thấp (hoặc Không MaxSim)

**Điều kiện:**
- `is_all_scanned = False`
- `len(text_bearing_idx) <= max_pages` (ít trang text)
- `lc_ratio < 0.4` HOẶC `doc_visual_emb is None`

**Cơ chế:**
```
top_indices = sorted(text_bearing_idx)    # tất cả trang text
```
Nếu `text_bearing_idx` rỗng (edge case: PDF có trang nhưng text cực ngắn):
```
top_indices = list(range(num_pages))     # fallback: tất cả trang
```

**Output:**
- `top_indices` = tất cả trang text (1–7 trang)
- `retrieval_tag` = `all_text(Np)`
- `document_text` = full text của tất cả trang
- `retrieved_pages = None` trong results.json khi chọn hết toàn bộ trang (is_all_pages=True)

**Ví dụ tiêu biểu:**
- PDF 4 trang, 0 scanned → `top_indices = [0,1,2,3]`, model nhận full document
- PDF 6 trang, 1 scanned, lc_ratio=17% → text_bearing_idx=5 pages, chỉ lấy 5 trang đó

---

### Case D — Text Nhiều, Hybrid Retrieval

**Điều kiện:**
- `is_all_scanned = False`
- `len(text_bearing_idx) > max_pages` (nhiều trang text, >7)

**Cơ chế:**
```
Bước 1: BM25 ranking
   → Tokenize tất cả trang + câu hỏi
   → Tokenizer: "vietnamese" nếu language=="vi", else cfg.retriever.tokenizer
   → _BM25Plus.get_scores(query_tokens)
   → Rank tất cả trang giảm dần

Bước 2: Dense E5 ranking
   → Load doc_emb từ cache (hoặc encode online nếu không có)
   → FAISS IndexFlatIP cosine similarity
   → Rank tất cả trang giảm dần

Bước 3: RRF Fusion
   → score(page) = Σ 1/(k=60 + rank_i) cho mỗi ranker
   → Trả về (page_idx, rrf_score) sorted giảm dần
   → Cũng trả về fused_scores dict để dùng ở evidence expansion

Bước 4: Adaptive Prune
   → tau = threshold_ratio (0.8) × max_score
   → Luôn lấy min_pages (3) đầu tiên
   → Tiếp tục lấy nếu score >= tau VÀ len < max_pages (7)
   → Re-sort theo thứ tự trang (ascending)

Bước 5: Hard cap assertion
   → assert len(top_indices) <= max_pages
```

**Output:**
- `top_indices` = 3–7 trang liên quan nhất
- `retrieval_tag` = `hybrid(bm25+dense) top=N/Mp`
- `fused_scores` = dict {page_idx: rrf_score} — **chỉ case này có scores**
- `document_text` = text của top pages, token-aware budget (≤28,000 tokens)

**Điểm đặc biệt:**
- Vietnamese: `tokenizer_name = "vietnamese"` → dùng underthesea hoặc trigrams
- Japanese: `tokenizer_name = "mecab"` (config mặc định) → MeCab morphological analysis
- `fused_scores` là dữ liệu duy nhất cho evidence expansion (Issue #4)

---

## Giai Đoạn 2 — VLM Images

Sau khi có `selected` (danh sách trang từ Retrieval), pipeline quyết định ảnh nào gửi
kèm prompt. Chỉ chạy khi `cfg.vlm.enabled = true`.

### Nhánh A — is_all_scanned

```
top_img_page_nums = sorted(all_pages[i]["page_num"] for i in top_indices[:max_imgs])

→ Render ảnh (96 DPI) cho các trang cần thiết
→ page_images_for_vlm = [img cho mỗi trang trong top_img_page_nums]

→ Prepend vào document_text:
   "[Scanned document. Image-to-page mapping: Image 1=Page X, Image 2=Page Y, ...]"
```

**Tại sao cần label:** Không có `=== Page N ===` text nào có nội dung. Model không thể
biết ảnh thứ 3 ứng với trang nào nếu không được chỉ rõ. Thiếu label → model không thể
điền `evidence_pages` chính xác.

### Nhánh B/C/D — Có Text (not all_scanned)

```
Render ảnh cho tất cả selected pages (để có sẵn trong cache)

Loop qua selected pages theo thứ tự:
  should_feed = (
      ("is_low_content" in feed_image_when AND page.is_low_content)
      OR
      ("has_table" in feed_image_when AND page.has_table)   ← luôn False hiện tại
  )
  if should_feed AND len(images) < max_imgs:
      feed ảnh của trang đó
```

**Hệ quả theo case:**

| Case | Điều kiện feed | Ảnh thực tế |
|---|---|---|
| **B** (mostly scan) | Scanned pages từ MaxSim có `is_low_content=True` | 1–4 ảnh (scanned pages) |
| **C** (text ít) | Trang text có `is_low_content=False` → không feed | Thường 0 ảnh |
| **D** (text nhiều) | Trang text: không feed. Trang scan trong top-7: feed | 0–3 ảnh tùy retrieved set |

**Điểm mù hiện tại:** `has_table` luôn là `False` (được set cứng khi parse).
Trang chứa bảng phức tạp nhưng `is_low_content=False` (text đủ dài) → không được
gửi ảnh → model chỉ đọc text extract (có thể mất cấu trúc bảng).

---

## Tổng Hợp: 6 Profile Input Đầy Đủ

### Profile 1 — Toàn scan, MaxSim OK `[A1]`

```
document_text:
  [Scanned document. Image-to-page mapping: Image 1=Page 3, Image 2=Page 8, ...]

  === Page 3 ===

  === Page 8 ===
  ...

page_images_for_vlm: [img_p3, img_p8, img_p12, img_p16, img_p20, img_p22]  ← max 6
fused_scores: {}
```

**Model nhận:** Label + page markers rỗng + 6 ảnh.
**Phụ thuộc hoàn toàn vào ảnh** để trả lời.

---

### Profile 2 — Toàn scan, không MaxSim `[A2]`

```
document_text:
  [Scanned document. Image-to-page mapping: Image 1=Page 1, Image 2=Page 2, ...]

  === Page 1 ===

  === Page 2 ===
  ...

page_images_for_vlm: [img_p1, img_p2, img_p3, img_p4, img_p5, img_p6]
fused_scores: {}
```

**Model nhận:** Ảnh 6 trang đầu — không có gì đảm bảo chứa câu trả lời.
**Chất lượng thấp nhất.**

---

### Profile 3 — Mostly scan, MaxSim bổ sung `[B]`

```
document_text:
  === Page 9 ===
  第15条　医療機関は...  ← text thật từ text-bearing page

  === Page 14 ===
                        ← rỗng (scanned page từ MaxSim)
  === Page 16 ===
                        ← rỗng

page_images_for_vlm: [img_p14, img_p16, img_p11, img_p3]  ← chỉ is_low_content pages
fused_scores: {}
```

**Model nhận:** Ít text + ảnh của trang scan liên quan nhất.
**Kết hợp text + ảnh để tìm câu trả lời.**

---

### Profile 4 — Text ít, lc thấp `[C]`

```
document_text:
  === Page 1 ===
  平成28年度事業報告書...  ← full text

  === Page 2 ===
  第1章 総論...

  ...  ← tất cả trang (thường 2–7 trang)

page_images_for_vlm: []  ← không có ảnh (all pages là is_low_content=False)
fused_scores: {}
```

**Model nhận:** Full text document, không có ảnh.
**Tốt nhất khi tài liệu ngắn và chứa đủ text.**

---

### Profile 5 — Text nhiều, tiếng Nhật `[D, tokenizer=mecab]`

```
document_text:
  === Page 9 ===
  [text của trang 9 - top ranked by BM25+Dense]

  === Page 10 ===
  [text của trang 10]

  ... (3–7 trang tổng)

page_images_for_vlm: [img_p14]  ← chỉ nếu trang 14 có is_low_content=True trong top-7
fused_scores: {8: 0.0323, 9: 0.0318, 13: 0.0305, ...}  ← dùng cho evidence expansion
```

**Model nhận:** Text của top pages từ hybrid retrieval + ảnh nếu có trang scan.

---

### Profile 6 — Text nhiều, tiếng Việt `[D, tokenizer=vietnamese]`

```
Giống Profile 5, nhưng tokenizer khác:
  BM25 dùng underthesea word_tokenize (hoặc whitespace + trigrams)
  → "đầu tư nước ngoài" được tokenize thành multi-word tokens
  → BM25 matching chính xác hơn cho tiếng Việt

document_text: text của top-7 pages tiếng Việt
page_images_for_vlm: [ảnh nếu có is_low_content page trong top-7]
fused_scores: {page_idx: score, ...}
```

---

## Tóm Tắt Ma Trận Quyết Định

```
                      lc_ratio
                0%      1–39%    40–99%    100%
              ┌───────┬────────┬─────────┬──────────┐
text_pages    │       │        │         │          │
≤ max_pages   │   C   │   C    │  B / C* │    –     │
              ├───────┼────────┼─────────┼──────────┤
text_pages    │       │        │         │          │
> max_pages   │   D   │   D    │   D**   │    –     │
              ├───────┼────────┼─────────┼──────────┤
is_all_scanned│   –   │   –    │    –    │ A1 / A2  │
              └───────┴────────┴─────────┴──────────┘

*  B khi MaxSim có, C khi MaxSim không có
** D: scanned pages được BM25+Dense rank gần 0 → ít khi được chọn
   → nếu câu trả lời nằm trong scanned page → retrieval miss → có thể zero answer
```

---

## Điểm Mù Đã Được Fix (Phase 2)

### 1. Case D + scanned pages → Issue #10 ✓ Fixed

```
Trước: PDF 70 trang, 4 trang pure-scan (< 50 chars)
  → Hybrid retrieval chọn top-7 từ 66 text pages
  → 4 trang scan bị rank = 0, không được chọn
  → Answer trong scan page → zero answer

Sau: Nếu còn slot trống (adaptive_prune trả về < 7 pages) AND MaxSim có:
  → MaxSim rank các pure-scanned pages chưa được chọn
  → Bổ sung vào top_indices, ghi "+scan(Np)" vào retrieval_tag
```

### 2. `has_table` luôn False → Issue #9 ✓ Fixed

```
Trước: has_table = False cứng → trang bảng không có ảnh gửi VLM

Sau: extract_pages() gọi page_obj.find_tables().tables
  → has_table = True khi phát hiện bảng
  → Trang có bảng nhưng is_low_content=False được gửi ảnh qua feed_image_when: [has_table]
```

### 3. Evidence Expansion chỉ ở Case D → Issue #11 ✓ Fixed

```
Trước: fused_scores = {} cho Cases A1, A2, B, C
  → expand_evidence không kích hoạt

Sau:
  Case A1: fused_scores = {page_idx: maxsim_score, ...} từ maxsim_score_list()
  Case B:  fused_scores = maxsim scores của cả MaxSim run (bổ sung sau mixed logic)
  Case A2, C: vẫn không có scores → expansion bỏ qua
```

### 4. `SCANNED_CHAR_THRESHOLD` không dùng → Issue #8 ✓ Fixed

```
Trước: is_all_scanned = num_low_content == num_pages (< 200 chars)
  → PDF all-image-dominant (50-199 chars/trang) bị coi là all-scanned
  → Bỏ qua BM25/Dense dù còn text thực sự

Sau: is_all_scanned = num_pure_scanned == num_pages (< 50 chars)
  → Chỉ PDF hoàn toàn không có text layer mới dùng MaxSim-only
  → text_bearing_idx dùng not is_pure_scanned → 50-199 char pages tham gia BM25
```

---

## Tóm Tắt Phân Tầng Ngưỡng Hiện Tại

| Chars | `is_pure_scanned` | `is_low_content` | Xử lý |
|-------|:-----------------:|:----------------:|--------|
| 0–49  | True | True | Loại khỏi BM25; MaxSim supplement; feed ảnh VLM |
| 50–199 | False | True | Tham gia BM25/Dense; feed ảnh VLM |
| ≥200 | False | False | BM25/Dense; feed ảnh chỉ khi `has_table=True` |

---

*Tài liệu được cập nhật sau Phase 2 fixes — `LAVA2026/INFERENCE_CASES.md`*
