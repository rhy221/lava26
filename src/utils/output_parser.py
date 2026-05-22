import json
import re


def parse_model_output(raw_output: str, answer_format: str) -> tuple[str, list[int]]:
    """
    Parse JSON output from the model into (answer_str, evidence_pages).

    The prompt asks for raw JSON, so json.loads() should usually succeed.
    The regex fallback handles: transformers backend, HTTP errors (empty string),
    or rare malformed/truncated responses.
    """
    json_str = _extract_json_block(raw_output)
    try:
        data = json.loads(json_str)
        raw_answer = data.get("answer", "")
        raw_pages = data.get("evidence_pages", [])
        return _format_answer(raw_answer, answer_format), _parse_pages(raw_pages)
    except (json.JSONDecodeError, ValueError):
        pass

    # Regex fallback for malformed or truncated JSON.
    answer = _regex_extract_answer(raw_output)
    pages = _regex_extract_pages(raw_output)
    if answer is not None:
        print(f"[WARN] JSON fallback. "
              f"answer={repr(answer)[:80]}")
        return _format_answer(answer, answer_format), pages

    if raw_output.strip():
        print(f"[WARN] Parse failed entirely. Raw ({len(raw_output)}c): "
              f"{raw_output[:200]!r}")
    return "0", [1]


def _regex_extract_answer(text: str):
    m = re.search(r'"answer"\s*:\s*(\[.*?\])', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            pass

    m = re.search(r'"answer"\s*:\s*(-?\d+(?:\.\d+)?)', text)
    if m:
        val = m.group(1)
        return float(val) if "." in val else int(val)

    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"?', text)
    if m:
        return m.group(1)

    return None


def _regex_extract_pages(text: str) -> list[int]:
    m = re.search(r'"evidence_pages"\s*:\s*(\[[\d,\s]*)', text)
    if m:
        nums = re.findall(r"\d+", m.group(1))
        return [int(n) for n in nums]
    return []


def _extract_json_block(text: str) -> str:
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    all_matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL))
    if all_matches:
        return all_matches[-1].group(0)
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return text.strip()


def _format_answer(raw_answer, answer_format: str) -> str:
    if answer_format in ("string", "number"):
        if isinstance(raw_answer, list):
            raw_answer = next((x for x in raw_answer if str(x).strip()), "0")
        return str(raw_answer).strip() or "0"

    if answer_format in ("unordered_list", "ordered_list"):
        if isinstance(raw_answer, list):
            items = raw_answer
        elif isinstance(raw_answer, str):
            try:
                parsed = json.loads(raw_answer.replace("'", '"'))
                items = parsed if isinstance(parsed, list) else [raw_answer]
            except (json.JSONDecodeError, ValueError):
                items = [raw_answer]
        else:
            items = [str(raw_answer)]
        if not items:
            items = ["0"]
        return "[" + ", ".join(f"'{str(item).strip()}'" for item in items) + "]"

    return str(raw_answer).strip()


def _parse_pages(raw_pages) -> list[int]:
    if isinstance(raw_pages, list):
        return [int(p) for p in raw_pages if str(p).isdigit() or isinstance(p, (int, float))]
    if isinstance(raw_pages, (int, float)):
        return [int(raw_pages)]
    if isinstance(raw_pages, str):
        return [int(n) for n in re.findall(r"\d+", raw_pages)]
    return []


def format_evidence_pages_for_csv(pages: list[int]) -> str:
    if not pages:
        return "[1]"
    return "[" + ", ".join(str(p) for p in sorted(set(pages))) + "]"
