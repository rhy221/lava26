"""ColQwen visual page retrieval + PDF page renderer.

Supports both ColQwen2.5 (colpali-engine>=0.3, transformers>=5.3) and
ColQwen2 (colpali-engine<0.3, transformers<5 / vLLM-compatible).

Enabled via config:
    visual.enabled: true
    visual.retriever_model: vidore/colqwen2-v1.0        # or colqwen2.5-v0.2
    visual.retriever_dpi: 120
    visual.retriever_batch_size: 2
"""
import torch


def _load_colqwen(model_name: str):
    """Load the best available ColQwen variant.

    Tries ColQwen2_5 first (colpali-engine>=0.3); falls back to ColQwen2
    (colpali-engine<0.3, compatible with transformers<5 / vLLM env).
    Returns (model, processor, variant_name).
    """
    try:
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        return ColQwen2_5, ColQwen2_5_Processor, "ColQwen2.5"
    except ImportError:
        pass
    try:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        return ColQwen2, ColQwen2Processor, "ColQwen2"
    except ImportError:
        raise ImportError(
            "colpali-engine not installed or missing ColQwen2/ColQwen2_5. "
            "Install: pip install 'colpali-engine<0.3.0'"
        )


def render_pdf_pages(pdf_path: str, dpi: int = 120) -> dict:
    """Render each PDF page to a PIL Image using pymupdf.

    Returns {1-indexed page_num: PIL.Image}. Used for both ColQwen encoding
    and VLM image feeding.
    """
    import fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    images = {}
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=dpi)
        images[i] = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return images


def render_pdf_pages_selective(pdf_path: str, page_nums: list, dpi: int = 120) -> dict:
    """Render only the specified (1-indexed) pages of a PDF.

    Much faster than render_pdf_pages() for large PDFs when only a few pages are needed.
    Returns {page_num: PIL.Image} for requested pages only.
    """
    import fitz
    from PIL import Image

    wanted = set(page_nums)
    doc = fitz.open(pdf_path)
    images = {}
    for i, page in enumerate(doc, start=1):
        if i in wanted:
            pix = page.get_pixmap(dpi=dpi)
            images[i] = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if len(images) == len(wanted):
            break
    doc.close()
    return images


class VisualRetriever:
    """ColQwen late-interaction visual retriever (ColPali style).

    Auto-detects ColQwen2.5 vs ColQwen2 based on what's installed.
    Recommended model per variant:
      ColQwen2.5 → vidore/colqwen2.5-v0.2   (colpali-engine>=0.3)
      ColQwen2   → vidore/colqwen2-v1.0      (colpali-engine<0.3, vLLM-safe)
    """

    def __init__(self, model_name: str = "vidore/colqwen2-v1.0", batch_size: int = 2):
        ModelClass, ProcessorClass, variant = _load_colqwen(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {variant} ({model_name}) on {device} ...")
        self.model = ModelClass.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self.processor  = ProcessorClass.from_pretrained(model_name)
        self.batch_size = batch_size
        print(f"{variant} ready.")

    @torch.no_grad()
    def encode_pages(self, page_images: list) -> list:
        """Encode PIL Images → list of batch tensors (batch_size_i, seq_len_i, dim).

        Images within a batch are padded to the same seq_len by the processor.
        Different batches can have different seq_lens (variable page resolutions),
        so we store batches separately instead of concatenating across batches.
        Tensors are kept on CPU to free GPU memory between batches.
        """
        all_batches = []
        for i in range(0, len(page_images), self.batch_size):
            batch = page_images[i : i + self.batch_size]
            inputs = self.processor.process_images(batch).to(self.model.device)
            emb = self.model(**inputs)  # (batch_size_i, padded_seq_len_i, dim)
            all_batches.append(emb.cpu())
        return all_batches

    @torch.no_grad()
    def ranked(self, question: str, doc_embeds: list) -> list:
        """Return 0-indexed page positions ranked by MaxSim score (descending).

        doc_embeds: list[Tensor] of shape (batch_size_i, seq_len_i, dim).
        Scores are aggregated across batches, then argsorted.
        """
        if not doc_embeds:
            return []
        q_inputs = self.processor.process_queries([question]).to(self.model.device)
        q_emb = self.model(**q_inputs)  # (1, q_seq_len, dim)

        all_scores = []
        for batch_emb in doc_embeds:
            # s: (1, batch_size_i) — one score per page in the batch
            s = self.processor.score_multi_vector(q_emb, batch_emb.to(self.model.device))
            all_scores.extend(s[0].tolist())

        return sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)


def maxsim_ranked(q_emb: "torch.Tensor", doc_emb_batches: list) -> list:
    """MaxSim ranking without the ColQwen model — used in Phase 2 inference.

    q_emb: Tensor (1, q_seq_len, dim) — pre-computed query embedding from Phase 1.
    doc_emb_batches: list[Tensor (batch_size, doc_seq_len, dim)] — from cache/visual/*.pt.
    Returns: 0-indexed page positions ranked by MaxSim score descending.
    """
    q = q_emb.squeeze(0).float()   # (q_len, dim)
    scores = []
    for batch in doc_emb_batches:
        b = batch.float()                                         # (batch_size, doc_len, dim)
        sim = torch.einsum('qd,bpd->qbp', q, b)                  # (q_len, batch_size, doc_len)
        score = sim.max(dim=-1).values.sum(dim=0)                 # (batch_size,)
        scores.extend(score.tolist())
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
