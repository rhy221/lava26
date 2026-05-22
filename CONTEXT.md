# LAVA 2026 - Project Context

## Competition

- Document VQA over Japanese/Vietnamese PDFs.
- Input: PDF + question. Output: answer + evidence page numbers.
- Answer formats: `string`, `number`, `unordered_list`, `ordered_list`.
- Metric: `Overall = (VQA Score + Grounding Score) / 2`.
- Test set: 624 questions, 2h limit.

## Repo

- Local: `c:\Users\Ngo Minh Tri\workspace\uit\LAVA2026`
- Server: `/datastore/$USER/LAVA2026`
- Dataset: `lava-challenge-2026/`

```text
config.yaml                   # source-of-truth config
run.py                        # Phase 1 preprocess + Phase 2 inference
src/config.py                 # OmegaConf loader, config_resolved.yaml snapshot
src/evaluate.py               # scoring helpers
src/utils/pdf_utils.py        # PyMuPDF extraction, token-aware context builder
src/utils/output_parser.py    # JSON/parser fallback + CSV formatting
src/utils/retriever.py        # BM25, dense E5, RRF, pruning
src/utils/visual_retriever.py # ColQwen cache + page rendering
scripts/job_lava.slurm        # server flow: preprocess -> vLLM -> inference
cache/                        # pages, dense, visual, query_embs.pt
runs/<JOB_ID>/                # results, submission, summary, logs, config snapshot
```

## Pipeline

```text
Phase 1: run.py --preprocess
  - parse PDFs -> cache/pages/<file_id>.json
  - dense E5 embeddings -> cache/dense/<file_id>.npy
  - optional ColQwen page/query embeddings -> cache/visual/, cache/query_embs.pt

Phase 2: run.py
  - load caches lazily
  - retrieve text pages with BM25 + dense E5 -> RRF -> adaptive pruning
  - build token-aware prompt text
  - optionally render/send VLM images
  - parse JSON answer and write Kaggle CSV
```

Important: current `run.py` does **not** use ColQwen ranking in text-page RRF. ColQwen cache is used for all-scanned PDFs to pick image pages for VLM input.

## Config

Use `config.yaml` plus dotlist overrides:

```bash
python run.py --config config.yaml data.split=train data.sample=20
python run.py --config config.yaml retriever.max_pages=10 retriever.tokenizer=mecab
python run.py --config config.yaml visual.enabled=false vlm.enabled=false
```

Every run writes `config_resolved.yaml`.

## Server

```bash
sbatch scripts/job_lava.slurm
OVERRIDES="data.split=train data.sample=20" sbatch scripts/job_lava.slurm
tail -f runs/<JOB_ID>/job.out
tail -f runs/<JOB_ID>/vllm_server.log
```

Assumed paths:

```text
PROJECT_DIR=/datastore/$USER/LAVA2026
PYTHON=/datastore/$USER/miniconda3/envs/vqa-jv/bin/python
VLLM=/datastore/$USER/miniconda3/envs/vqa-jv/bin/vllm
MODEL_DIR=/datastore/$USER/models/Qwen3.6-27B
COLQWEN_DIR=/datastore/$USER/models/colqwen2.5-v0.2
```

## Current Commit Suggestion

Suggested split:

- `Add config-driven two-phase LAVA inference pipeline`
- `Add ColQwen cache and VLM image selection`
- `Update LAVA docs for current config and SLURM flow`

Single-commit option:

```text
Document and wire config-driven two-phase LAVA VLM pipeline
```
