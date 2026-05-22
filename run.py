"""
LAVA 2026 — Two-phase inference pipeline (config-driven via OmegaConf)

Phase 1 (--preprocess):
    Parse ALL unique PDFs with pymupdf → disk cache.
    Encode dense embeddings (e5-large) and visual embeddings (ColQwen) → disk cache.
    Runs BEFORE the vLLM / Qwen server starts (full GPU for ColQwen).

Phase 2 (inference):
    Load all caches from disk — near-zero per-PDF overhead.
    Retrieve top text pages via BM25 + Dense (RRF fusion).
    Use cached ColQwen MaxSim only to pick image pages for all-scanned PDFs.
    Call Qwen 27B VLM for final answer.

Usage:
    python run.py [--config config.yaml] [--output runs/ID/results.json] [--preprocess]
                  [--clear-cache]        # wipe cache before Phase 1 (accurate timing)
                  [key=value ...]        # OmegaConf dotlist overrides
"""

import argparse
import ast
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config, snapshot
from src.evaluate import aggregate_results, grounding_score, overall_score, vqa_score
from src.utils.output_parser import format_evidence_pages_for_csv, parse_model_output
from src.utils.pdf_utils import build_document_text_token_aware, extract_pages
from src.utils.retriever import DenseRetriever, hybrid_top_pages

try:
    from src.utils.visual_retriever import (
        VisualRetriever, render_pdf_pages, render_pdf_pages_selective, maxsim_ranked,
    )
except ImportError:
    VisualRetriever = None  # type: ignore[assignment,misc]
    def render_pdf_pages(pdf_path, dpi=120): return {}  # type: ignore[misc]
    def render_pdf_pages_selective(pdf_path, page_nums, dpi=120): return {}  # type: ignore[misc]
    def maxsim_ranked(q_emb, doc_emb_batches): return []  # type: ignore[misc]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    bar = "=" * 60
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{bar}\n  [{ts}] {text}\n{bar}")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _pil_to_b64(img, max_side: int = 1008) -> str:
    """Resize longest side to ≤ max_side (divisible by Qwen patch=28), encode as base64 PNG."""
    import base64
    from io import BytesIO
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


# ── Transformers backend ──────────────────────────────────────────────────────

def load_model_transformers(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[Model] Loading {model_path} (4-bit NF4) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=quant_cfg, device_map="auto", trust_remote_code=True,
    ).eval()
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[Model] VRAM: {alloc:.1f} GB / {total:.1f} GB")
    return model, tokenizer


def infer_transformers(model, tokenizer, document_text, question, answer_format, language, cfg):
    import torch
    user_content = cfg.prompts.user_template.format(
        document_text=document_text, language=language,
        question=question, answer_format=answer_format,
    )
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": cfg.prompts.system},
         {"role": "user",   "content": user_content}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=cfg.model.qwen.enable_thinking,
    )
    max_new_tokens = cfg.generation.max_new_tokens.get(answer_format, 160)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return result


# ── vLLM backend ──────────────────────────────────────────────────────────────

def _answer_schema(answer_format: str) -> dict:
    """JSON schema helper kept for possible vLLM guided decoding experiments."""
    if answer_format == "number":
        answer_type: dict = {"type": "number"}
    elif answer_format in ("unordered_list", "ordered_list"):
        answer_type = {"type": "array", "items": {"type": "string"}, "minItems": 1}
    else:
        answer_type = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "answer": answer_type,
            "evidence_pages": {
                "type": "array", "items": {"type": "integer"}, "minItems": 1,
            },
        },
        "required": ["answer", "evidence_pages"],
        "additionalProperties": False,
    }


def wait_for_vllm(base_url: str, timeout: int = 600) -> None:
    import urllib.error, urllib.request
    print(f"[vLLM] Waiting for server at {base_url} ...")
    t0 = time.time()
    deadline = t0 + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base_url}/models", timeout=2)
            elapsed = time.time() - t0
            print(f"[vLLM] Server ready  ({elapsed:.0f}s to start).")
            return
        except (urllib.error.URLError, OSError):
            time.sleep(3)
    raise RuntimeError(f"[vLLM] Server at {base_url} did not respond within {timeout}s")


def infer_vllm(base_url, model_name, document_text, question,
               answer_format, language, cfg, page_images=None):
    """POST to vLLM /chat/completions; prompt asks for raw JSON output."""
    import urllib.error, urllib.request

    user_text = cfg.prompts.user_template.format(
        document_text=document_text, language=language,
        question=question, answer_format=answer_format,
    )
    max_tokens = cfg.generation.max_new_tokens.get(answer_format, 160)

    if page_images:
        user_content = [{"type": "text", "text": user_text}]
        for img in page_images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_pil_to_b64(img)}"},
            })
    else:
        user_content = user_text

    payload = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": cfg.prompts.system},
            {"role": "user",   "content": user_content},
        ],
        "chat_template_kwargs": {"enable_thinking": cfg.model.qwen.enable_thinking},
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        usage = data.get("usage", {})
        tokens_str = (f"tokens={usage.get('prompt_tokens',0)}/"
                      f"{usage.get('completion_tokens',0)}" if usage else "")
        return data["choices"][0]["message"]["content"], tokens_str
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        print(f"    [WARN] vLLM HTTP {e.code}: {body}")
        return "", ""
    except Exception as e:
        print(f"    [WARN] vLLM request failed: {type(e).__name__}: {e}")
        return "", ""


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_dirs(cfg):
    root = Path(cfg.parsing.cache_dir).parent
    return {
        "pages":  Path(cfg.parsing.cache_dir),
        "dense":  root / "dense",
        "visual": root / "visual",
    }


def _load_parsed_pages(file_id, pdf_path, dirs, cfg):
    """Load parsed pages: disk cache → pymupdf. Returns (pages: list[dict], source: str)."""
    cache_file = dirs["pages"] / f"{file_id}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8")), "cache"
        except Exception:
            pass

    raw = extract_pages(str(pdf_path))
    pages = [{"page_num": p, "text": t, "is_low_content": lc, "has_table": False}
             for p, t, lc in raw]
    if pages:
        cache_file.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return pages, "pymupdf"


def _load_dense_emb(file_id, page_texts, dense_retriever, dirs):
    """Load dense embeddings: disk cache → online encode. Returns (emb, source)."""
    emb_path = dirs["dense"] / f"{file_id}.npy"
    if emb_path.exists():
        return np.load(str(emb_path)), "cache"
    emb = dense_retriever.encode_pages(page_texts)
    return emb, "online"


# ── Phase 1: Preprocessing ────────────────────────────────────────────────────

def run_preprocessing(df, pdf_dir, cfg, dense_retriever, clear_cache: bool = False):
    """Parse + encode all unique PDFs → disk cache (runs before vLLM server starts).

    Step 1: pymupdf parse + dense encode  (fast, CPU-only)
    Step 2: Load ColQwen → visual page encode → query encode → unload ColQwen
    """
    dirs = _cache_dirs(cfg)

    # ── Optional cache wipe for clean-slate timing ────────────────────────────
    if clear_cache:
        print(f"[{_ts()}] [Cache] Clearing all cache dirs ...")
        for key, d in dirs.items():
            if d.exists():
                shutil.rmtree(d)
                print(f"  Deleted {key}: {d}")
        q_emb_path = dirs["visual"].parent / "query_embs.pt"
        if q_emb_path.exists():
            q_emb_path.unlink()
            print(f"  Deleted: {q_emb_path}")
        print()

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    unique_ids = df["file_id"].unique()
    _banner(f"PHASE 1 — Preprocessing  ({len(unique_ids)} unique PDFs, split={cfg.data.split})")

    n_cached = sum(1 for fid in unique_ids if (dirs["pages"] / f"{fid}.json").exists())
    n_vis_cached = sum(1 for fid in unique_ids if (dirs["visual"] / f"{fid}.pt").exists())
    query_emb_path = dirs["visual"].parent / "query_embs.pt"
    print(f"[{_ts()}] Cache status:")
    print(f"  pages  : {n_cached}/{len(unique_ids)} cached")
    print(f"  visual : {n_vis_cached}/{len(unique_ids)} cached")
    print(f"  query  : {'yes' if query_emb_path.exists() else 'no'}")
    print()

    t0 = time.time()
    all_parsed: dict = {}

    # ── Step 1: Parse + Dense ─────────────────────────────────────────────────
    n_to_parse = len(unique_ids) - n_cached
    print(f"[{_ts()}] Step 1 — pymupdf parse + dense encode "
          f"({n_to_parse} new, {n_cached} cached) ...")
    t_step1 = time.time()
    for file_id in tqdm(unique_ids, desc="Parse+Dense", ncols=80):
        pdf_path = pdf_dir / f"{file_id}.pdf"
        if not pdf_path.exists():
            print(f"  [WARN] PDF not found: {pdf_path}")
            continue

        t_pdf = time.time()
        t_p = time.time()
        pages, parse_src = _load_parsed_pages(file_id, pdf_path, dirs, cfg)
        t_parse = time.time() - t_p

        if not pages:
            print(f"  [WARN] No pages for {file_id} — skipping.")
            continue

        all_parsed[file_id] = pages

        # Skip dense encode for fully-scanned PDFs (empty text → useless embeddings)
        n_lc = sum(1 for p in pages if p["is_low_content"])
        is_fully_scanned = n_lc == len(pages)

        t_d = time.time()
        dense_skipped = True
        if dense_retriever is not None and not is_fully_scanned \
                and not (dirs["dense"] / f"{file_id}.npy").exists():
            emb = dense_retriever.encode_pages([p["text"] for p in pages])
            np.save(str(dirs["dense"] / f"{file_id}.npy"), emb)
            dense_skipped = False
        t_dense = time.time() - t_d

        elapsed = time.time() - t_pdf
        dense_tag = "skip(scanned)" if (is_fully_scanned and dense_retriever is not None) \
                    else f"{t_dense:.1f}s"
        if parse_src == "pymupdf" or t_dense > 0.5:
            print(f"  [{file_id}] {len(pages)}p ({n_lc} scanned) | "
                  f"parse={t_parse:.1f}s ({parse_src}) | dense={dense_tag} | total={elapsed:.1f}s")

    step1_elapsed = time.time() - t_step1
    n_parsed = len(all_parsed)
    print(f"[{_ts()}] Step 1 done — {n_parsed} PDFs in memory | elapsed={step1_elapsed:.0f}s\n")

    # ── Step 2: Load ColQwen → visual page encode + query encode ─────────────
    visual_retriever = None
    if cfg.visual.enabled:
        if VisualRetriever is None:
            print(f"[{_ts()}] [ColQwen] WARNING: colpali-engine not installed — skipping.")
        else:
            try:
                print(f"[{_ts()}] [ColQwen] Loading {cfg.visual.retriever_model} ...")
                t_load = time.time()
                visual_retriever = VisualRetriever(
                    model_name=cfg.visual.retriever_model,
                    batch_size=cfg.visual.retriever_batch_size,
                )
                print(f"[{_ts()}] [ColQwen] Ready  ({time.time()-t_load:.0f}s to load).")
            except Exception as e:
                print(f"[{_ts()}] [ColQwen] WARNING: load failed ({e}) — skipping.")

    if visual_retriever is not None:
        # ── Step 2a: Page encode ──────────────────────────────────────────────
        n_to_encode = sum(
            1 for fid in unique_ids
            if fid in all_parsed and not (dirs["visual"] / f"{fid}.pt").exists()
        )
        print(f"[{_ts()}] Step 2a — ColQwen page encode "
              f"({n_to_encode} new, {n_vis_cached} cached) ...")
        def _vram_str() -> str:
            if not torch.cuda.is_available():
                return ""
            alloc = torch.cuda.memory_allocated(0) / 1024 ** 3
            reserved = torch.cuda.memory_reserved(0) / 1024 ** 3
            total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return f"VRAM alloc={alloc:.1f}G reserved={reserved:.1f}G total={total:.0f}G"

        t_step2a = time.time()
        n_encoded = 0
        n_failed  = 0
        for file_id in tqdm(unique_ids, desc="Page encode", ncols=80):
            pdf_path = pdf_dir / f"{file_id}.pdf"
            if not pdf_path.exists() or file_id not in all_parsed:
                continue
            if (dirs["visual"] / f"{file_id}.pt").exists():
                continue
            pages = all_parsed[file_id]
            t_v = time.time()
            try:
                imgs    = render_pdf_pages(str(pdf_path), dpi=cfg.visual.retriever_dpi)
                max_enc = min(len(pages), getattr(cfg.visual, "max_encode_pages", len(pages)))
                idxs    = list(range(max_enc))
                pil_list = [imgs[pages[i]["page_num"]] for i in idxs if pages[i]["page_num"] in imgs]
                embs = visual_retriever.encode_pages(pil_list)
                torch.save({"embs": embs, "enc_indices": idxs[:len(pil_list)]},
                           str(dirs["visual"] / f"{file_id}.pt"))
                n_encoded += 1
                vram = _vram_str()
                print(f"  [{file_id}] {time.time()-t_v:.1f}s ({len(pil_list)} pages encoded)"
                      + (f" | {vram}" if vram else ""), flush=True)
            except Exception as exc:
                n_failed += 1
                print(f"  [{file_id}] ERROR — {type(exc).__name__}: {exc} | {_vram_str()}", flush=True)
                import traceback; traceback.print_exc()
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        step2a_elapsed = time.time() - t_step2a
        print(f"[{_ts()}] Step 2a done — {n_encoded} encoded, {n_failed} failed"
              f" | elapsed={step2a_elapsed:.0f}s | {_vram_str()}\n", flush=True)

        # ── Step 2b: Query encode ─────────────────────────────────────────────
        if not query_emb_path.exists():
            print(f"[{_ts()}] Step 2b — ColQwen query encode ({len(df)} queries) ...")
            t_step2b = time.time()
            query_embs: dict = {}
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Query embs", ncols=80):
                q_inputs = visual_retriever.processor.process_queries(
                    [row["question"]]
                ).to(visual_retriever.model.device)
                with torch.no_grad():
                    q_emb = visual_retriever.model(**q_inputs).cpu()
                query_embs[str(row["id"])] = q_emb
            torch.save(query_embs, str(query_emb_path))
            step2b_elapsed = time.time() - t_step2b
            rate = len(df) / step2b_elapsed
            print(f"[{_ts()}] Step 2b done — {len(df)} queries in {step2b_elapsed:.0f}s "
                  f"({rate:.1f} q/s) → {query_emb_path}")
        else:
            saved = torch.load(str(query_emb_path), weights_only=False)
            print(f"[{_ts()}] Step 2b — skipped (already cached: {len(saved)} query embs)")

    total = time.time() - t0
    n_done = sum(1 for fid in unique_ids if (dirs["pages"] / f"{fid}.json").exists())
    n_vis_done = sum(1 for fid in unique_ids if (dirs["visual"] / f"{fid}.pt").exists())
    _banner(f"PHASE 1 DONE — {total:.0f}s ({total/60:.1f} min)")
    print(f"  pages cached  : {n_done}/{len(unique_ids)}")
    print(f"  visual cached : {n_vis_done}/{len(unique_ids)}")
    print(f"  query embs    : {'yes' if query_emb_path.exists() else 'NO — visual retrieval will be disabled'}")
    print(f"  Est. Phase 2  : {len(df)} questions — see Phase 2 logs for timing")


# ── Phase 2: Inference ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LAVA 2026 — Inference Pipeline")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--output",      default=None,
                        help="Results JSON path (default: runs/<split>_results.json)")
    parser.add_argument("--preprocess",  action="store_true",
                        help="Phase 1: parse + encode all unique PDFs then exit.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Wipe cache before Phase 1 for accurate end-to-end timing.")
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf dotlist overrides, e.g. retriever.max_pages=10")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    data_dir  = Path(cfg.data.dir)
    split     = cfg.data.split
    out_path  = Path(args.output) if args.output else Path("runs") / f"{split}_results.json"
    run_dir   = out_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    snapshot(cfg, run_dir)

    # ── Load models ───────────────────────────────────────────────────────────
    model = tokenizer = None
    backend = cfg.model.qwen.backend
    if not args.preprocess:
        if backend == "transformers":
            model, tokenizer = load_model_transformers(cfg.model.qwen.path)
        else:
            wait_for_vllm(cfg.model.qwen.vllm_url)
            from transformers import AutoTokenizer
            print(f"[{_ts()}] [Tokenizer] Loading from {cfg.model.qwen.path} ...")
            tokenizer = AutoTokenizer.from_pretrained(cfg.model.qwen.path, trust_remote_code=True)
            print(f"[{_ts()}] [Tokenizer] Ready.")

    # ── Dense retriever ───────────────────────────────────────────────────────
    dense_retriever = None
    if str(cfg.retriever.dense_model).lower() not in ("none", "0", ""):
        print(f"[{_ts()}] [Dense] Loading {cfg.retriever.dense_model} ...")
        dense_retriever = DenseRetriever(cfg.retriever.dense_model)
        print(f"[{_ts()}] [Dense] Ready.")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(data_dir / f"{split}.csv")
    if cfg.data.sample:
        df = df.head(cfg.data.sample)
    pdf_dir = data_dir / f"{split}_pdfs"
    dirs    = _cache_dirs(cfg)

    # ── Phase 1: Preprocessing ────────────────────────────────────────────────
    if args.preprocess:
        run_preprocessing(df, pdf_dir, cfg, dense_retriever,
                          clear_cache=args.clear_cache)
        return

    # ── Phase 2: Inference ────────────────────────────────────────────────────
    unique_ids = df["file_id"].unique()
    n_cached   = sum(1 for fid in unique_ids if (dirs["pages"] / f"{fid}.json").exists())

    # Load pre-computed query embeddings (no ColQwen model needed in Phase 2)
    query_emb_path = dirs["visual"].parent / "query_embs.pt"
    precomp_query_embs: dict = {}
    if query_emb_path.exists():
        precomp_query_embs = torch.load(str(query_emb_path), weights_only=False)
        print(f"[{_ts()}] [Setup] Pre-computed ColQwen query embs: "
              f"{len(precomp_query_embs)} loaded")

    retriever_tag = "+".join(filter(None, [
        "bm25",
        "dense" if dense_retriever else None,
    ]))
    vlm_tag = f"vlm({cfg.vlm.max_images_per_prompt}img)" if cfg.vlm.enabled else "text"

    _banner(f"PHASE 2 — Inference  [{split} | {backend} | {retriever_tag} | {vlm_tag}]")
    print(f"[{_ts()}] Questions  : {len(df)} from {split}.csv")
    print(f"[{_ts()}] Cache warm : {n_cached}/{len(unique_ids)} PDFs pre-parsed"
          + (f"  ({len(unique_ids)-n_cached} will parse on-demand)"
             if n_cached < len(unique_ids) else ""))
    print()

    # In-memory caches (populated lazily from disk)
    page_cache:       dict = {}
    emb_cache:        dict = {}
    visual_emb_cache: dict = {}
    page_img_cache:   dict = {}

    results:         list = []
    submission_rows: list = []
    t_phase2_start = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Questions", ncols=80):
        q_id          = row["id"]
        file_id       = row["file_id"]
        question      = row["question"]
        answer_format = row["answer_format"]
        language      = row["language"]

        pdf_path = pdf_dir / f"{file_id}.pdf"
        if not pdf_path.exists():
            print(f"[WARN] PDF not found: {pdf_path}")
            continue

        t_q = time.time()
        t_parse = t_dense = t_colqwen = t_render = 0.0
        parse_src = emb_src = vis_src = "-"

        # ── Parse (disk cache → pymupdf) ─────────────────────────────────
        if file_id not in page_cache:
            _t = time.time()
            pages, parse_src = _load_parsed_pages(file_id, pdf_path, dirs, cfg)
            page_cache[file_id] = pages
            t_parse = time.time() - _t

        all_pages = page_cache[file_id]
        num_pages       = len(all_pages)
        num_low_content = sum(1 for p in all_pages if p["is_low_content"])
        is_all_scanned  = num_low_content == num_pages
        page_texts      = [p["text"] for p in all_pages]

        # ── Dense embeddings (disk cache → online encode) ─────────────────
        doc_emb = None
        if dense_retriever is not None and not is_all_scanned:
            if file_id not in emb_cache:
                _t = time.time()
                emb_cache[file_id], emb_src = _load_dense_emb(
                    file_id, page_texts, dense_retriever, dirs)
                t_dense = time.time() - _t
            doc_emb = emb_cache[file_id]

        # ── ColQwen page embeddings (disk cache only — no model in Phase 2) ─
        doc_visual_emb = None
        if file_id not in visual_emb_cache:
            vis_path = dirs["visual"] / f"{file_id}.pt"
            if vis_path.exists():
                _t = time.time()
                data = torch.load(str(vis_path), weights_only=False)
                visual_emb_cache[file_id] = (data["embs"], data["enc_indices"])
                vis_src = "cache"
                t_colqwen = time.time() - _t
        doc_visual_emb = visual_emb_cache.get(file_id)

        # ── Page retrieval ────────────────────────────────────────────────
        # ColQwen MaxSim is NOT used for text-PDF retrieval (net negative vs
        # BM25+Dense baseline). For all-scanned PDFs, MaxSim drives both page
        # selection and image selection (Issue #6 fix).
        text_bearing_idx = [i for i, p in enumerate(all_pages) if not p["is_low_content"]]
        fused_scores: dict[int, float] = {}  # populated by hybrid path; used for Issue #4

        # Issue #2 fix: select tokenizer based on document language
        tokenizer_name = "vietnamese" if language == "vi" else cfg.retriever.tokenizer

        if is_all_scanned:
            # Issue #6 fix: use MaxSim to pick the most relevant pages so
            # retrieved_pages is non-empty and grounding is meaningful.
            # Previously all pages were "selected" → retrieved_pages = null.
            q_emb_for_ret = precomp_query_embs.get(str(q_id))
            if q_emb_for_ret is not None and doc_visual_emb is not None:
                embs, enc_indices = doc_visual_emb
                local_ranked = maxsim_ranked(q_emb_for_ret, embs)
                top_indices = sorted(
                    enc_indices[i] for i in local_ranked[:cfg.retriever.max_pages]
                    if i < len(enc_indices) and enc_indices[i] < num_pages
                )
                retrieval_tag = f"maxsim_visual({len(top_indices)}p)"
            else:
                top_indices = list(range(min(cfg.retriever.max_pages, num_pages)))
                retrieval_tag = f"seq_fallback({len(top_indices)}p)"

        elif len(text_bearing_idx) <= cfg.retriever.max_pages:
            top_indices   = sorted(text_bearing_idx) if text_bearing_idx else list(range(num_pages))
            retrieval_tag = f"all_text({len(top_indices)}p)"

        else:
            # Issue #3 fix: removed effective_max inflation (+3/+1) that caused
            # 62% of cases to exceed the configured max_pages=7 cap.
            top_indices, fused_scores = hybrid_top_pages(
                question, page_texts, dense_retriever,
                min_pages=cfg.retriever.min_pages,
                max_pages=cfg.retriever.max_pages,
                doc_emb=doc_emb,
                tokenizer_name=tokenizer_name,
                rrf_k=cfg.retriever.rrf_k,
                threshold_ratio=cfg.retriever.threshold_ratio,
            )
            # Issue #3: hard cap assertion — catches regressions in retriever logic
            assert len(top_indices) <= cfg.retriever.max_pages, (
                f"[BUG] Retrieval cap violated: got {len(top_indices)} > "
                f"{cfg.retriever.max_pages} pages"
            )
            retrieval_tag = f"hybrid(bm25+dense) top={len(top_indices)}/{num_pages}p"

        selected      = [all_pages[i] for i in top_indices]
        selected_nums = [p["page_num"] for p in selected]
        is_all_pages  = sorted(selected_nums) == list(range(1, num_pages + 1))

        document_text = build_document_text_token_aware(
            selected, tokenizer, max_tokens=cfg.generation.max_prompt_tokens
        )

        # ── VLM images ────────────────────────────────────────────────────
        page_images_for_vlm: list = []
        vlm_img_source = ""
        if cfg.vlm.enabled and selected:
            if file_id not in page_img_cache:
                page_img_cache[file_id] = {}
            page_imgs = page_img_cache[file_id]

            max_imgs = cfg.vlm.max_images_per_prompt

            if is_all_scanned:
                # Issue #6 fix: reuse the top_indices already selected by MaxSim
                # (retrieval section above). This ensures retrieved_pages in
                # results.json and the images sent to VLM are the same pages.
                # Previously MaxSim was re-run here independently, so retrieved_pages
                # could differ from the images the model actually sees.
                top_img_page_nums = sorted(
                    all_pages[i]["page_num"] for i in top_indices[:max_imgs]
                )
                vlm_img_source = retrieval_tag.split("(")[0]  # "maxsim_visual" or "seq_fallback"

                needed = [pn for pn in top_img_page_nums if pn not in page_imgs]
                if needed:
                    _t = time.time()
                    page_imgs.update(
                        render_pdf_pages_selective(str(pdf_path), needed, dpi=cfg.vlm.image_dpi)
                    )
                    t_render = time.time() - _t

                for pn in top_img_page_nums:
                    if pn in page_imgs:
                        page_images_for_vlm.append(page_imgs[pn])

                # Issue #6 fix: prepend image→page mapping so the model can ground
                # evidence_pages correctly (it otherwise can't tell which image = which page)
                if page_images_for_vlm:
                    labels = ", ".join(
                        f"Image {idx+1}=Page {pn}"
                        for idx, pn in enumerate(top_img_page_nums[:len(page_images_for_vlm)])
                    )
                    document_text = (
                        f"[Scanned document. Image-to-page mapping: {labels}. "
                        f"Use these page numbers in evidence_pages.]\n\n" + document_text
                    )
            else:
                # Text-bearing PDF: render selected pages, feed only low_content or has_table
                needed = [p["page_num"] for p in selected if p["page_num"] not in page_imgs]
                if needed:
                    _t = time.time()
                    page_imgs.update(
                        render_pdf_pages_selective(str(pdf_path), needed, dpi=cfg.vlm.image_dpi)
                    )
                    t_render = time.time() - _t

                for p in selected:
                    if len(page_images_for_vlm) >= max_imgs:
                        break
                    should_feed = (
                        any((c == "is_low_content" and p["is_low_content"]) or
                            (c == "has_table" and p.get("has_table", False))
                            for c in cfg.vlm.feed_image_when)
                        and p["page_num"] in page_imgs
                    )
                    if should_feed:
                        page_images_for_vlm.append(page_imgs[p["page_num"]])

        # ── Inference ─────────────────────────────────────────────────────
        t_infer_start = time.time()
        tokens_str = ""
        if backend == "transformers":
            raw_output = infer_transformers(
                model, tokenizer, document_text, question, answer_format, language, cfg)
        else:
            raw_output, tokens_str = infer_vllm(
                cfg.model.qwen.vllm_url, cfg.model.qwen.path,
                document_text, question, answer_format, language, cfg,
                page_images=page_images_for_vlm or None,
            )
        t_infer = time.time() - t_infer_start
        elapsed = time.time() - t_q

        pred_answer, pred_pages = parse_model_output(raw_output, answer_format)

        # Issue #4 fix: expand evidence_pages when model under-reports.
        # When exactly 1 page is predicted AND retrieval found closely-scored
        # top-2 pages (RRF score gap ≤ expand_evidence_score_diff), include
        # both. This lifts Dice grounding score for multi-page answers.
        pp_cfg = getattr(cfg, "post_process", None)
        if (pp_cfg and getattr(pp_cfg, "expand_evidence", False)
                and len(pred_pages) == 1
                and len(selected_nums) >= 3
                and fused_scores):
            # Map 1-indexed page numbers → 0-indexed to look up fused_scores
            pn_to_idx = {all_pages[i]["page_num"]: i for i in range(num_pages)}
            by_score = sorted(
                selected_nums,
                key=lambda pn: fused_scores.get(pn_to_idx.get(pn, -1), 0.0),
                reverse=True,
            )
            if len(by_score) >= 2:
                s1 = fused_scores.get(pn_to_idx.get(by_score[0], -1), 0.0)
                s2 = fused_scores.get(pn_to_idx.get(by_score[1], -1), 0.0)
                diff_ratio = (s1 - s2) / max(s1, 1e-12)
                if diff_ratio <= pp_cfg.expand_evidence_score_diff:
                    extra = by_score[1]
                    if extra not in pred_pages:
                        pred_pages = sorted(pred_pages + [extra])

        evidence_csv = format_evidence_pages_for_csv(pred_pages)

        submission_rows.append({"id": q_id, "answer": pred_answer,
                                 "evidence_page_number": evidence_csv})

        result = {
            "id": q_id, "file_id": file_id, "question": question,
            "answer_format": answer_format, "language": language,
            "num_pages": num_pages, "num_low_content": num_low_content,
            "retrieved_pages": None if is_all_pages else selected_nums,
            "predicted_answer": pred_answer, "predicted_pages": pred_pages,
            "raw_model_output": raw_output[:500],
            "time_seconds": round(elapsed, 2),
        }

        # ── Per-question log ──────────────────────────────────────────────
        pages_tag   = f"all({num_pages}p)" if is_all_pages else str(selected_nums)
        scanned_tag = " [all_scanned]" if is_all_scanned else ""
        imgs_tag    = (f" | imgs={len(page_images_for_vlm)}({vlm_img_source})"
                       if page_images_for_vlm else "")
        src_tag     = f" | src={parse_src}"

        timing_parts = []
        if t_parse   > 0.05: timing_parts.append(f"parse={t_parse:.1f}s({parse_src})")
        if t_dense   > 0.05: timing_parts.append(f"dense={t_dense:.1f}s({emb_src})")
        if t_colqwen > 0.05: timing_parts.append(f"vis={t_colqwen:.1f}s({vis_src})")
        if t_render  > 0.05: timing_parts.append(f"render={t_render:.1f}s")
        timing_parts.append(f"infer={t_infer:.1f}s")
        if tokens_str:        timing_parts.append(tokens_str)

        if split == "train":
            gt_answer = str(row["answer"])
            try:
                gt_pages = [int(p) for p in ast.literal_eval(str(row["evidence_page_number"]))]
                if not isinstance(gt_pages, list): gt_pages = [gt_pages]
            except (ValueError, SyntaxError):
                gt_pages = []

            vqa_s     = vqa_score(pred_answer, gt_answer, answer_format)
            grounding = grounding_score(pred_pages, gt_pages)
            overall   = overall_score(vqa_s, grounding)
            missed    = (selected_nums is not None and gt_pages
                         and not any(p in selected_nums for p in gt_pages))
            result.update({
                "ground_truth_answer": gt_answer, "ground_truth_pages": gt_pages,
                "vqa_score":      round(vqa_s,     4),
                "grounding_score": round(grounding, 4),
                "overall_score":  round(overall,   4),
                "retriever_missed": missed,
            })
            miss_tag = " [MISS]" if missed else ""
            print(f"  [{q_id}] {elapsed:.1f}s | {pages_tag}{scanned_tag}{imgs_tag}{src_tag}"
                  f" | VQA={vqa_s:.2f} Ground={grounding:.2f} Overall={overall:.2f}{miss_tag}")
        else:
            fail_tag = ""
            if pred_answer in ("0", ""):
                if is_all_scanned:                                      fail_tag = " [all_scanned]"
                elif all(p["is_low_content"] for p in selected):        fail_tag = " [image_dom]"
                else:                                                   fail_tag = " [zero_ans]"
            print(f"  [{q_id}] {elapsed:.1f}s | {pages_tag}{scanned_tag}{imgs_tag}"
                  f"{src_tag}{fail_tag}")

        # Always print retrieval + timing detail
        print(f"    [retrieval] {retrieval_tag}")
        print(f"    [timing]   {' | '.join(timing_parts)}")

        results.append(result)

        # ── Progress summary every 50 questions ───────────────────────────
        n_done = len(results)
        if n_done % 50 == 0:
            elapsed_total = time.time() - t_phase2_start
            avg_so_far = elapsed_total / n_done
            remaining  = len(df) - n_done
            eta_sec    = avg_so_far * remaining
            zeros_so_far = sum(1 for r in results if r.get("predicted_answer", "") in ("0", ""))
            print(f"\n[{_ts()}] ── Progress: {n_done}/{len(df)} "
                  f"| avg={avg_so_far:.1f}s/q "
                  f"| zeros={zeros_so_far}/{n_done} ({zeros_so_far/n_done*100:.0f}%) "
                  f"| ETA≈{eta_sec/60:.0f}min\n")

    # ── Save outputs ──────────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[{_ts()}] [Output] Results → {out_path}")

    sub_path = run_dir / f"{split}_submission.csv"
    pd.DataFrame(submission_rows).to_csv(sub_path, index=False, encoding="utf-8-sig")
    print(f"[{_ts()}] [Output] Submission CSV → {sub_path}")

    if not results:
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    times    = [r["time_seconds"] for r in results]
    total    = sum(times)
    avg      = total / len(times)
    p50      = sorted(times)[len(times) // 2]
    p95      = sorted(times)[int(len(times) * 0.95)]
    failures = sum(1 for r in results if r.get("predicted_answer", "") in ("0", ""))

    # Parse failure detection.
    parse_fails = sum(1 for r in results
                      if r.get("raw_model_output", "").startswith("{") is False
                      and r.get("predicted_answer", "") in ("0", ""))

    lc_pages = sum(r.get("num_low_content", 0) for r in results)
    all_pgs  = sum(r.get("num_pages", 0) for r in results)

    _banner(f"SUMMARY  [{split.upper()} | {backend} | {retriever_tag}]")
    print(f"  Questions         : {len(results)}")
    print(f"  Zero/fail answers : {failures}  ({failures/len(results)*100:.1f}%)")
    if all_pgs:
        print(f"  Low-content pages : {lc_pages}/{all_pgs}  ({lc_pages/all_pgs*100:.1f}%)")
    print(f"  Avg time/question : {avg:.1f}s")
    print(f"  Median / P95      : {p50:.1f}s / {p95:.1f}s")
    print(f"  Total Phase 2     : {total:.0f}s  ({total/60:.1f} min)")
    print(f"  Est. 624q @ {avg:.1f}s : {avg*624/3600:.2f} h")

    if split == "train":
        train_results = [r for r in results if "vqa_score" in r]
        if train_results:
            agg = aggregate_results(train_results)
            print(f"  Mean VQA          : {agg['mean_vqa_score']:.4f}")
            print(f"  Mean Grounding    : {agg['mean_grounding_score']:.4f}")
            print(f"  Mean Overall      : {agg['mean_overall_score']:.4f}")
            with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(agg, f, ensure_ascii=False, indent=2)
    else:
        agg = {
            "num_questions": len(results), "zero_answers": failures,
            "avg_sec": round(avg, 2), "median_sec": round(p50, 2), "p95_sec": round(p95, 2),
            "total_sec": round(total, 2), "est_624q_hours": round(avg * 624 / 3600, 2),
            "backend": backend, "retriever": retriever_tag,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"[{_ts()}] [Output] summary.json → {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
