# LAVA 2026 — Phân Tích Chi Tiết Pipeline

> **Mục đích tài liệu này:** Tổng hợp toàn bộ luồng xử lý, kiến trúc, và các lựa chọn kỹ thuật của hệ thống Document VQA cho cuộc thi LAVA Challenge 2026.

---

## 1. Tổng Quan Dự Án

LAVA 2026 là hệ thống **Visual Question Answering (VQA) đa ngôn ngữ** xử lý tài liệu PDF tiếng Nhật và tiếng Việt.

| Thành phần | Mô tả |
|---|---|
| **Input** | PDF tài liệu + câu hỏi ngôn ngữ tự nhiên |
| **Output** | Câu trả lời (string/number/list) + số trang bằng chứng |
| **Metric** | `(VQA Score + Grounding Score) / 2` |
| **Ngôn ngữ** | Tiếng Nhật, Tiếng Việt |
| **Giới hạn thời gian** | 2 giờ cho ~624 câu hỏi |

---

## 2. Kiến Trúc Tổng Thể

Pipeline chia làm **2 pha độc lập** để tối ưu sử dụng GPU:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        config.yaml (source of truth)                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
        ▼                                               ▼
┌───────────────────┐                       ┌──────────────────────┐
│   PHA 1 (CPU+GPU) │                       │  PHA 2 (vLLM Server) │
│   Preprocessing   │                       │  LLM Inference       │
│                   │                       │                      │
│  PyMuPDF          │                       │  BM25 + Dense RRF    │
│  E5-large         │──── cache/ ──────────▶│  Token-aware context │
│  ColQwen2.5       │                       │  Qwen3.6-27B VLM     │
└───────────────────┘                       └──────────────────────┘
                                                        │
                                                        ▼
                                            ┌──────────────────────┐
                                            │  runs/{JOB_ID}/      │
                                            │  submission.csv      │
                                            │  results.json        │
                                            │  summary.json        │
                                            └──────────────────────┘
```

---

## 3. Cấu Trúc Thư Mục

```
LAVA2026/
├── run.py                      # Entry point chính (839 dòng)
├── config.yaml                 # Cấu hình tập trung (OmegaConf)
├── requirements.txt            # Dependencies
├── src/
│   ├── config.py               # OmegaConf loader, config snapshot
│   ├── evaluate.py             # VQA/grounding scoring metrics
│   └── utils/
│       ├── pdf_utils.py        # PyMuPDF, token budgeting
│       ├── retriever.py        # BM25, E5-large, RRF fusion
│       ├── output_parser.py    # JSON parsing, CSV formatting
│       └── visual_retriever.py # ColQwen, MaxSim ranking
├── scripts/
│   └── job_lava.slurm          # SLURM cluster submission
├── notebooks/
│   └── analysis.ipynb          # Phân tích kết quả chạy
├── cache/                      # (gitignored) Embeddings đã precompute
└── runs/                       # (gitignored) Kết quả thực nghiệm
```

---

## 4. Hệ Thống Cấu Hình

Toàn bộ pipeline được điều khiển bởi `config.yaml` thông qua **OmegaConf**, hỗ trợ override từ CLI:

```bash
# Ví dụ override từ command line
python run.py --config config.yaml data.split=train retriever.max_pages=10
python run.py --config config.yaml visual.enabled=false vlm.enabled=false
```

### Các nhóm cấu hình chính:

| Nhóm | Tham số quan trọng | Mô tả |
|---|---|---|
| `model.qwen` | `path`, `backend` (vllm/transformers), `mode` | Mô hình sinh câu trả lời |
| `data` | `dir`, `split`, `sample` | Dữ liệu đầu vào |
| `parsing` | `engine`, `cache_dir`, `max_page_chars` (8000) | Trích xuất PDF |
| `retriever` | `dense_model`, `tokenizer`, `min_pages` (3), `max_pages` (7), `rrf_k` (60) | Tìm kiếm trang |
| `visual` | `enabled`, `retriever_model`, `retriever_dpi` (120), `max_encode_pages` (40) | ColQwen visual |
| `vlm` | `enabled`, `image_dpi` (96), `feed_image_when`, `max_images_per_prompt` (6) | Ảnh input VLM |
| `generation` | `max_new_tokens` (theo loại), `max_prompt_tokens` (28000) | Sinh text |

---

## 5. Pha 1 — Tiền Xử Lý (Preprocessing)

**Chạy trước** khi khởi động vLLM server. Mục đích: xây dựng cache trên disk để loại bỏ overhead khi inference.

### Bước 1.1: Phân tích trang PDF

```
PDF files → PyMuPDF → per-page JSON → cache/pages/{file_id}.json
```

**Chi tiết xử lý** (`src/utils/pdf_utils.py`):
- Dùng `extract_pages()` để trích xuất từng trang riêng biệt
- Mỗi trang giới hạn tối đa **8,000 ký tự**
- Ưu tiên chế độ markdown (giữ cấu trúc bảng), fallback về plain text
- Đánh dấu `is_low_content = True` nếu text < 200 ký tự (trang scan/ảnh)
- Bỏ qua dense encoding cho PDF hoàn toàn là ảnh (không có text layer)

### Bước 1.2: Dense Embeddings (E5-large)

```
pages text → SentenceTransformer (E5-large-multilingual) → cache/dense/{file_id}.npy
```

- Encode với prefix `"passage: "` theo chuẩn E5
- Dùng FAISS IndexFlatIP cho similarity search sau này
- **Bỏ qua** nếu PDF toàn scan (không có text để encode)

### Bước 1.3: Visual Encoding (ColQwen2.5)

```
PDF pages → render tại 120 DPI → PIL Images → ColQwen encode → cache/visual/{file_id}.pt
                                                              → cache/query_embs.pt
```

**Chi tiết** (`src/utils/visual_retriever.py`):
- Render tất cả trang thành ảnh (DPI=120)
- Giới hạn **40 trang/PDF** để kiểm soát VRAM
- Encode dùng **late-interaction embeddings** (ColQwen)
- Pre-compute **tất cả câu hỏi** thành `query_embs.pt` cho Pha 2
- ColQwen được **unload sau Pha 1** để giải phóng GPU cho vLLM

---

## 6. Pha 2 — Inference

Xử lý từng câu hỏi theo pipeline 5 bước. Không load ColQwen (dùng cache).

### Bước 2.1: Truy Xuất Trang (Hybrid Retrieval)

```
question + pages → BM25 + Dense E5 → RRF Fusion → Adaptive Prune → top 3-7 pages
```

#### 6.1.1 Tokenization

Hai chiến lược tùy ngôn ngữ (`src/utils/retriever.py`):

| Chiến lược | Kích hoạt | Cơ chế |
|---|---|---|
| **Bigram** (mặc định) | Tiếng Việt, Latin | CJK character bigrams + whitespace split cho Latin/Việt |
| **MeCab** | Tiếng Nhật | Phân tích hình thái học + bigrams/trigrams, lọc trợ từ Nhật |

#### 6.1.2 BM25 Ranking

- Dùng thư viện `rank_bm25`
- Tính điểm BM25 cho tất cả trang trong PDF

#### 6.1.3 Dense E5 Ranking (nếu bật)

- Load cache embeddings từ disk
- FAISS similarity search (cosine)
- Prefix câu hỏi với `"query: "` theo chuẩn E5

#### 6.1.4 RRF Fusion

```
score(page) = Σ 1/(k + rank_i)   với k=60
```

Kết hợp ranking BM25 và Dense thành một điểm thống nhất.

#### 6.1.5 Adaptive Pruning

```python
threshold = 0.8 × max_score
keep pages where score >= threshold
enforce min_pages=3, max_pages=7
re-sort by page order (coherence)
```

**Trường hợp đặc biệt:**
- Tổng trang ≤ max_pages → dùng tất cả trang có text
- PDF toàn scan → dùng tất cả trang (không có text để retrieve)

---

### Bước 2.2: Xây Dựng Document Context (Token-Aware)

```
selected pages → format with markers → token budget check → document_text
```

**Định dạng output:**
```
=== Page 1 ===
[nội dung trang 1]

=== Page 5 ===
[nội dung trang 5]
```

**Token Budget** (`src/utils/pdf_utils.py: build_document_text_token_aware()`):

| Thành phần | Token |
|---|---|
| Tổng ngữ cảnh Qwen | 32,768 |
| Dự trữ cho prompt | 1,500 |
| Dự trữ cho output | 2,048 |
| **Còn lại cho document** | **~28,000** |

> **Lý do dùng tokenizer thực:** Heuristic chars/token sai cho CJK (1 ký tự Nhật/Trung ≈ 1-2 tokens, không phải 0.25 như Latin). Dùng `tokenizer.encode()` trực tiếp để chính xác.

---

### Bước 2.3: Chọn Ảnh Cho VLM

Hai trường hợp khác nhau:

#### Trường hợp A: PDF toàn scan (all-scanned)

```
query_emb (cache) + doc_emb (cache) → MaxSim scoring → top 6 pages → PIL Images
```

**MaxSim** (`src/utils/visual_retriever.py: maxsim_ranked()`):
```
score(page) = max over query_tokens [ max over page_tokens (q · p) ]
```
Không cần load ColQwen — chỉ dùng embeddings đã cache.

Fallback: 6 trang đầu nếu không có MaxSim cache.

#### Trường hợp B: PDF có text

```
selected pages → render tại 96 DPI → lọc: is_low_content OR has_table → tối đa 6 ảnh
```

Chỉ đưa ảnh vào khi trang khó đọc bằng text thuần (bảng, biểu đồ, ảnh scan lẫn).

**Encoding:** PIL Image → resize ≤1008px → base64 PNG (Qwen patch size=28)

---

### Bước 2.4: LLM Inference

#### Backend vLLM (production)

```
POST http://localhost:{port}/chat/completions
{
  "model": "Qwen3.6-27B",
  "messages": [system_msg, user_msg_with_images],
  "max_tokens": <per answer type>
}
```

Cấu hình vLLM server:
- GPU memory utilization: 50% (40GB trên A100 80GB)
- 4-bit NF4 quantization (bitsandbytes)
- enforce-eager mode
- Tối đa 6 ảnh/prompt

#### Backend Transformers (fallback/local)

- BitsAndBytes 4-bit NF4 quantization
- Chat template với thinking disabled (`enable_thinking=False`)
- Device map auto

#### Prompt System

```
[SYSTEM]
Bạn là chuyên gia phân tích tài liệu. Chỉ trả về JSON thuần, không giải thích.

[USER]
Document:
=== Page 1 ===
...

Question: {question}

Format rules:
- number: {"answer": "42", "evidence_pages": [1, 5]}
- string: {"answer": "...", "evidence_pages": [2]}
- unordered_list: {"answer": ["item1", "item2"], "evidence_pages": [3]}
- Nếu không tìm thấy: answer = "0" hoặc ["0"]
```

**max_new_tokens theo loại câu trả lời:**

| Loại | Tokens |
|---|---|
| number | 96 |
| string | 160 |
| ordered_list | 384 |
| unordered_list | 512 |

---

### Bước 2.5: Phân Tích Output & Định Dạng

**Parser** (`src/utils/output_parser.py`):

```
raw LLM output
    │
    ├─ json.loads() ──────────────────── OK → extract answer + evidence_pages
    │
    ├─ regex extract JSON block ──────── OK → extract answer + evidence_pages
    │   (code fences, nested braces)
    │
    └─ _regex_extract_answer() ───────── fallback → "0" / ["0"]
```

**Định dạng answer:**
- string/number → plain string
- list → Python list string: `['item1', 'item2']`

**Evidence pages:**
- Parse integers từ JSON array
- Default `[1]` nếu rỗng
- CSV format: `[1, 5, 8]`

---

## 7. Output & Kết Quả

Mỗi lần chạy tạo thư mục `runs/{JOB_ID}/`:

| File | Nội dung |
|---|---|
| `submission.csv` | Format nộp Kaggle: question_id, answer, evidence_pages |
| `results.json` | Chi tiết từng câu hỏi: input, output, timing, scores |
| `summary.json` | Thống kê tổng: VQA score, grounding score, timing |
| `config_resolved.yaml` | Snapshot config để tái hiện kết quả |

---

## 8. Scoring & Evaluation

**`src/evaluate.py`:**

| Metric | Cách tính |
|---|---|
| **VQA Score** | String normalization match, hoặc F1 cho list, hoặc LCS cho ordered list |
| **Grounding Score** | Dice coefficient: `2×|A∩B| / (|A|+|B|)` trên tập trang bằng chứng |
| **Overall Score** | `(VQA Score + Grounding Score) / 2` |

---

## 9. Triển Khai HPC (SLURM)

**`scripts/job_lava.slurm`** — Workflow 2 pha tự động:

```bash
# Tài nguyên
#SBATCH --mem=120G
#SBATCH --gres=gpu:1 (A100 80GB)
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00

# Phase 1: Preprocessing
python run.py --preprocess --config config.yaml

# Start vLLM server (port động)
PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
vllm serve Qwen3.6-27B --port $PORT --quantization bitsandbytes ...

# Phase 2: Inference
python run.py --config config.yaml model.qwen.backend=vllm ...

# Cleanup
kill $VLLM_PID
```

**Lý do tách 2 pha:**
- ColQwen2.5 chiếm VRAM đáng kể trong Pha 1
- Sau Pha 1, ColQwen được unload hoàn toàn
- Qwen3.6-27B trong Pha 2 nhận được 40GB VRAM sạch (50% × 80GB)

---

## 10. Các Quyết Định Kỹ Thuật Quan Trọng

### 10.1 Tại sao RRF thay vì weighted sum?

RRF (Reciprocal Rank Fusion) không cần calibrate weights giữa BM25 và Dense — hai hệ thống có score range khác nhau (BM25 không bị giới hạn, cosine trong [-1,1]). RRF chỉ dùng ranking, loại bỏ vấn đề này.

### 10.2 Tại sao ColQwen chỉ dùng cho all-scanned PDFs?

Chi phí MaxSim `O(Q × P × D_q × D_p)` cao hơn text retrieval. Với PDF có text, BM25+Dense đã đủ mạnh. ColQwen chỉ bù khi không có text để retrieve.

### 10.3 Tại sao token budgeting dùng tokenizer thực?

Heuristic 4 chars/token sai nghiêm trọng cho CJK: 1 kanji = 1 token, nhưng = 3 bytes UTF-8. Dùng `tokenizer.encode()` trực tiếp cho phép nhét đúng số trang vào context window mà không tràn.

### 10.4 Tại sao adaptive pruning thay vì fixed top-k?

Tài liệu ngắn (10 trang) vs dài (200 trang) cần số trang evidence khác nhau. Threshold `0.8 × max_score` tự động điều chỉnh, min_pages=3 đảm bảo ngữ cảnh tối thiểu.

### 10.5 Tại sao precompute query embeddings?

Mỗi câu hỏi cần encoded ColQwen embedding để MaxSim. Precompute 1 lần trong Pha 1 (khi ColQwen còn loaded) thay vì reload ColQwen trong Pha 2 tiết kiệm ~5-10 phút.

---

## 11. Hiệu Năng Thực Tế

Từ test run (JOB 23933, 624 câu hỏi, A100 80GB):

| Metric | Giá trị |
|---|---|
| Thời gian Pha 1 | ~12 phút |
| Thời gian Pha 2 | ~27 phút |
| **Tổng** | **~39 phút << 2 giờ** |
| Tốc độ inference | 2.57 giây/câu hỏi (trung bình) |
| P95 latency | 5.38 giây |
| Zero answers | 24.8% (155/624) — chủ yếu PDF scan |

---

## 12. Dependencies Chính

| Thư viện | Phiên bản | Mục đích |
|---|---|---|
| `pymupdf` | ≥1.24.0 | Trích xuất PDF |
| `torch` + `transformers` | ≥2.3.0 / ≥4.51.0 | LLM backbone |
| `vllm` | ≥0.8.0 | Inference server |
| `rank_bm25` | ≥0.2.2 | BM25 retrieval |
| `sentence-transformers` | ≥3.0.0 | E5-large dense encoding |
| `faiss-cpu` | ≥1.8.0 | Vector similarity search |
| `colpali-engine` | ≥0.3.0 | ColQwen visual retrieval |
| `omegaconf` | ≥2.3.0 | Config management |
| `fugashi[unidic-lite]` | ≥1.3.0 | MeCab Japanese tokenizer |
| `bitsandbytes` | ≥0.43.0 | 4-bit quantization |

---

## 13. Luồng Dữ Liệu Tổng Hợp

```
[PDF files] + [questions]
        │
        ▼
┌── PHA 1 ──────────────────────────────────────────────────────┐
│                                                               │
│  PyMuPDF extract_pages()                                      │
│      └─▶ cache/pages/{file_id}.json                          │
│                                                               │
│  E5-large encode_pages()                                      │
│      └─▶ cache/dense/{file_id}.npy                           │
│                                                               │
│  ColQwen render + encode pages                                │
│      └─▶ cache/visual/{file_id}.pt                           │
│                                                               │
│  ColQwen encode all questions                                  │
│      └─▶ cache/query_embs.pt                                 │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
   [ColQwen unloaded, vLLM server started]
        │
        ▼
┌── PHA 2 (per question) ───────────────────────────────────────┐
│                                                               │
│  1. RETRIEVAL                                                 │
│     BM25(question, pages)                                     │
│     + Dense E5(question, cached_emb)                         │
│     → RRF fusion (k=60)                                       │
│     → Adaptive prune (0.8×max, min=3, max=7)                 │
│                                                               │
│  2. CONTEXT BUILDING                                          │
│     selected_pages → "=== Page N ===" format                 │
│     → token_aware_budget (≤28,000 tokens)                    │
│                                                               │
│  3. IMAGE SELECTION                                           │
│     all-scanned: MaxSim(query_emb, visual_cache) → top 6    │
│     has-text: filter low_content/table pages → ≤6 images    │
│                                                               │
│  4. LLM INFERENCE                                             │
│     Qwen3.6-27B vLLM                                         │
│     input: [system_prompt, document_text, question, images]  │
│     output: {"answer": ..., "evidence_pages": [...]}         │
│                                                               │
│  5. PARSING & FORMATTING                                      │
│     json.loads() → regex fallback → "0" default             │
│     format answer + evidence_pages                           │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
[runs/{JOB_ID}/submission.csv + results.json + summary.json]
```

---

*Tài liệu được tạo tự động từ phân tích source code — `LAVA2026/PIPELINE_ANALYSIS.md`*
