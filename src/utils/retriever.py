import re

import numpy as np

try:
    from rank_bm25 import BM25Plus as _BM25
except ImportError:
    from rank_bm25 import BM25Okapi as _BM25  # fallback

try:
    import fugashi as _fugashi
    _MECAB_AVAILABLE = True
except ImportError:
    _MECAB_AVAILABLE = False

# Lazy MeCab tagger — instantiated once on first use
_mecab_tagger = None

_JP_PARTICLES = frozenset({
    "の","は","を","が","に","へ","で","と","から","まで","や","も",
    "て","ば","か","な","よ","ね","さ","し","も","だ","ら","ず","ん",
})


# ── Tokenizers ────────────────────────────────────────────────────────────────

def _tokenize_bigram(text: str) -> list[str]:
    """Character bigrams for CJK/kana; whitespace split for Latin/Vietnamese."""
    tokens = []
    cjk = re.compile(r'[぀-ヿ㐀-䶿一-鿿豈-﫿]')
    for chunk in re.split(r'([぀-ヿ㐀-䶿一-鿿豈-﫿]+)', text.lower()):
        if cjk.match(chunk):
            tokens.extend(chunk[i:i+2] for i in range(len(chunk) - 1)) if len(chunk) > 1 else tokens.append(chunk)
        else:
            tokens.extend(t for t in chunk.split() if t)
    return tokens or ["<empty>"]


def _tokenize_mecab(text: str) -> list[str]:
    """MeCab morphological tokenizer with bigrams + trigrams + tech terms.

    Falls back to bigram if fugashi is not installed.
    Reduces Japanese particle dominance and improves recall for compound terms.
    """
    global _mecab_tagger
    if not _MECAB_AVAILABLE:
        return _tokenize_bigram(text)

    if _mecab_tagger is None:
        _mecab_tagger = _fugashi.Tagger()

    words = [
        w.surface.lower()
        for w in _mecab_tagger(text)
        if w.surface.strip() and w.surface.lower() not in _JP_PARTICLES
    ]

    bigrams  = [f"{words[i]}_{words[i+1]}"          for i in range(len(words) - 1)]
    trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words) - 2)]
    tech     = re.findall(r'[A-Za-z]+\d+|\d+\.\d+|\d+[A-Za-z]+|[A-Z]{2,}', text)

    tokens = words + bigrams + trigrams + [t.lower() for t in tech]
    return tokens or ["<empty>"]


def _tokenize_vietnamese(text: str) -> list[str]:
    """Vietnamese: NFC-normalize + whitespace split + trigrams.
    Uses underthesea word_tokenize if available (better multi-syllable recall).
    Diacritics are preserved after NFC normalization — do NOT strip them.
    Why: Vietnamese uses Latin script with tonal diacritics (ổ, ướ …) separated
    by spaces. The generic bigram tokenizer works for CJK but misses Vietnamese
    compound words like "đầu tư" or "nước ngoài" that carry distinct meaning."""
    import unicodedata
    text = unicodedata.normalize("NFC", text.lower())
    try:
        from underthesea import word_tokenize as _vn_wt  # optional dep
        words = _vn_wt(text, format="text").split()
        return words or ["<empty>"]
    except (ImportError, Exception):
        pass
    # Option B fallback: whitespace words + character trigrams for compound coverage
    words = text.split()
    tris = [text[i:i+3] for i in range(max(0, len(text) - 2))
            if len(text[i:i+3].strip()) == 3]
    return (words + tris) or ["<empty>"]


def _tokenize(text: str, mode: str = "bigram") -> list[str]:
    """Dispatch to the configured tokenizer."""
    if mode == "mecab":
        return _tokenize_mecab(text)
    if mode == "vietnamese":
        return _tokenize_vietnamese(text)
    return _tokenize_bigram(text)


# ── BM25 (sparse) ─────────────────────────────────────────────────────────────

def bm25_top_k(question: str, page_texts: list[str], top_k: int,
               tokenizer_name: str = "bigram") -> list[int]:
    """Return sorted 0-indexed page indices ranked by BM25 (kept for compat)."""
    n = len(page_texts)
    if n <= top_k:
        return list(range(n))
    tokenized = [_tokenize(t, tokenizer_name) for t in page_texts]
    bm25 = _BM25(tokenized)
    scores = bm25.get_scores(_tokenize(question, tokenizer_name))
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return sorted(ranked[:top_k])


def _bm25_ranked(question: str, page_texts: list[str],
                 tokenizer_name: str = "bigram") -> list[int]:
    """All page indices ranked by BM25 score descending."""
    tokenized = [_tokenize(t, tokenizer_name) for t in page_texts]
    bm25 = _BM25(tokenized)
    scores = bm25.get_scores(_tokenize(question, tokenizer_name))
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


# ── Dense retriever (multilingual-e5-large) ───────────────────────────────────

class DenseRetriever:
    """
    Wraps multilingual-e5-large + FAISS for per-PDF dense page retrieval.

    encode_pages() is split from ranked() so callers can cache embeddings
    per file_id and avoid re-encoding the same PDF for every question.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DenseRetriever device: {device}")
        self.model = SentenceTransformer(model_name, device=device)

    def encode_pages(self, page_texts: list[str]) -> np.ndarray:
        passages = [f"passage: {t}" for t in page_texts]
        return self.model.encode(
            passages, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        ).astype(np.float32)

    def ranked(self, question: str, page_texts: list[str],
               doc_emb: "np.ndarray | None" = None) -> list[int]:
        import faiss
        if doc_emb is None:
            doc_emb = self.encode_pages(page_texts)
        q_emb = self.model.encode(
            [f"query: {question}"], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        n, d = doc_emb.shape
        index = faiss.IndexFlatIP(d)
        index.add(doc_emb)
        _, idxs = index.search(q_emb, n)
        return idxs[0].tolist()


# ── RRF fusion ────────────────────────────────────────────────────────────────

def rrf_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists of 0-indexed page indices."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, page_idx in enumerate(ranked, start=1):
            scores[page_idx] = scores.get(page_idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Adaptive pruning ──────────────────────────────────────────────────────────

def adaptive_prune(
    scored: list[tuple[int, float]],
    min_pages: int = 3,
    max_pages: int = 7,
    threshold_ratio: float = 0.8,
) -> list[int]:
    """HEAR-style relative thresholding with min/max caps.

    τ = threshold_ratio × max_score; pages above τ are kept.
    Always returns at least min_pages, never more than max_pages.
    Result re-sorted by original page order for coherent LLM context.
    """
    if not scored:
        return []
    max_score = scored[0][1]
    tau = threshold_ratio * max_score
    selected: list[int] = []
    for page_idx, score in scored:
        if len(selected) < min_pages or (score >= tau and len(selected) < max_pages):
            selected.append(page_idx)
        if len(selected) >= max_pages:
            break
    return sorted(selected)


# ── High-level hybrid entry point (3-way RRF: BM25 + Dense + Visual) ──────────

def hybrid_top_pages(
    question: str,
    page_texts: list[str],
    dense_retriever: "DenseRetriever | None",
    min_pages: int = 3,
    max_pages: int = 7,
    doc_emb: "np.ndarray | None" = None,
    precomputed_visual_ranked: "list[int] | None" = None,
    tokenizer_name: str = "bigram",
    rrf_k: int = 60,
    threshold_ratio: float = 0.8,
) -> tuple[list[int], dict[int, float]]:
    """RRF hybrid page retrieval: BM25 + dense, with optional visual ranking.

    Returns (sorted_page_indices, fused_score_dict) where fused_score_dict maps
    0-indexed page index → RRF score. Callers can use scores for post-processing
    (e.g. evidence_pages expansion when top-2 scores are close).

    Visual ranking is supplied as a pre-computed ranked list (computed in Phase 1
    via maxsim_ranked) so no ColQwen model is needed at inference time.
    Falls back gracefully when dense_retriever is None or precomputed_visual_ranked is None.

    Fix #1: was `n <= min_pages` — too narrow, caused empty retrieval for PDFs
    with min_pages < n <= max_pages. Now short-circuits for all small PDFs.
    Fix #1b: if adaptive_prune returns [] (all-zero/NaN scores), fall back to
    first min_pages indices so retrieved_pages is never empty.
    """
    n = len(page_texts)
    # Issue #1 fix: return all pages when PDF is small enough to skip retrieval
    if n <= max_pages:
        return list(range(n)), {}

    ranked_lists = [_bm25_ranked(question, page_texts, tokenizer_name)]

    if dense_retriever is not None:
        ranked_lists.append(dense_retriever.ranked(question, page_texts, doc_emb=doc_emb))

    if precomputed_visual_ranked is not None:
        ranked_lists.append(precomputed_visual_ranked)

    fused = rrf_fusion(ranked_lists, k=rrf_k)
    fused_score_dict = dict(fused)

    pruned = adaptive_prune(fused, min_pages=min_pages, max_pages=max_pages,
                            threshold_ratio=threshold_ratio)

    # Issue #1b fallback: prevent empty list when all scores are zero or NaN
    if not pruned:
        pruned = list(range(min(min_pages, n)))

    return pruned, fused_score_dict
