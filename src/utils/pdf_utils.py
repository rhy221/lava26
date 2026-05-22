import fitz  # pymupdf


PROMPT_OVERHEAD_TOKENS = 1_500  # system prompt + question + format instructions
OUTPUT_RESERVE_TOKENS  = 2_048  # must match MAX_NEW_TOKENS in run.py
CONTEXT_LIMIT_TOKENS   = 32_768
# chars/token = 1.5 (conservative: Japanese kanji can be <2 chars/token in dense text)
MAX_DOC_CHARS = int((CONTEXT_LIMIT_TOKENS - PROMPT_OVERHEAD_TOKENS - OUTPUT_RESERVE_TOKENS) * 1.5)


def extract_text_from_pdf(pdf_path: str, max_chars: int = MAX_DOC_CHARS) -> tuple[str, list[int]]:
    """
    Extract text from all pages of a PDF using pymupdf.
    Returns (document_text, pages_with_content).

    Each page is wrapped with "=== Page N ===" markers.
    Hard char limit calibrated for Japanese text (2 chars/token) to stay
    safely within the 32768-token context window.
    """
    doc = fitz.open(pdf_path)
    parts = []
    pages_with_content = []
    total_chars = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        page_label = page_num + 1

        page_block = f"=== Page {page_label} ===\n{text}\n"
        block_chars = len(page_block)

        if total_chars + block_chars > max_chars:
            parts.append(
                f"\n[NOTE: Document truncated at page {page_label - 1} due to length limit. "
                f"Total pages: {len(doc)}]\n"
            )
            break

        parts.append(page_block)
        if text:
            pages_with_content.append(page_label)
        total_chars += block_chars

    doc.close()
    return "\n".join(parts), pages_with_content


def get_pdf_page_count(pdf_path: str) -> int:
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


SCANNED_CHAR_THRESHOLD = 50    # truly no text layer (pure image/scanned page)
IMAGE_DOM_CHAR_THRESHOLD = 200  # sparse text; mostly visual (maps, flowcharts)
# Per-page char cap: prevents one extremely dense page from consuming entire budget
MAX_PAGE_CHARS = 8_000


def _extract_page_text(page) -> str:
    """
    Extract text from a single pymupdf page.
    Tries markdown mode first (preserves table structure, pymupdf>=1.24.3);
    falls back to plain text if markdown is unsupported or returns empty.
    """
    try:
        md = page.get_text("markdown").strip()
        if md:
            return md
    except (AssertionError, Exception):
        pass
    return page.get_text().strip()


def extract_pages(pdf_path: str) -> list[tuple[int, str, bool]]:
    """
    Return [(1-indexed page_num, text, is_low_content), ...] for all pages.

    is_low_content=True for both:
      - truly scanned pages (zero text layer)
      - image-dominant pages (maps, flowcharts — sparse labels only)

    Text is capped at MAX_PAGE_CHARS per page to prevent one dense page from
    consuming the entire context budget.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = _extract_page_text(doc[i])
        text = text[:MAX_PAGE_CHARS]  # hard cap per page
        is_low_content = len(text) < IMAGE_DOM_CHAR_THRESHOLD
        pages.append((i + 1, text, is_low_content))
    doc.close()
    return pages


def build_document_text(pages: list[tuple[int, str, bool]], max_chars: int = MAX_DOC_CHARS) -> str:
    """Build the === Page N === formatted string from a (page_num, text, is_scanned) list."""
    parts = []
    total_chars = 0
    for page_num, text, _is_scanned in pages:
        block = f"=== Page {page_num} ===\n{text}\n"
        if total_chars + len(block) > max_chars:
            parts.append(
                f"\n[NOTE: Truncated after page {page_num - 1} due to length limit.]\n"
            )
            break
        parts.append(block)
        total_chars += len(block)
    return "\n".join(parts)


def build_document_text_token_aware(
    pages: "list[tuple[int, str, bool]] | list[dict]",
    tokenizer,
    max_tokens: int = 28_000,
) -> str:
    """Build === Page N === string with token-accurate budget.

    Accepts both tuple format (page_num, text, is_low_content) and dict format
    {page_num, text, ...} so callers don't need to normalise before calling.

    Fixes HTTP 400 context overflow on dense Japanese text where the old
    1.5 chars/token heuristic underestimates actual token count by 2-3×.
    """
    parts, total = [], 0
    for page in pages:
        if isinstance(page, dict):
            page_num, text = page["page_num"], page["text"]
        else:
            page_num, text, _ = page
        block = f"=== Page {page_num} ===\n{text}\n"
        n = len(tokenizer.encode(block, add_special_tokens=False))
        if total + n > max_tokens:
            parts.append(
                f"\n[NOTE: Truncated after page {page_num - 1} due to token budget.]\n"
            )
            break
        parts.append(block)
        total += n
    return "\n".join(parts)
