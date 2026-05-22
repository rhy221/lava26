# LAVA 2026 - UIT Submission

Multi-page Document VQA over Japanese and Vietnamese PDFs for the LAVA Challenge 2026.

## Pipeline

```text
config.yaml + dotlist overrides
  -> Phase 1: python run.py --preprocess
     - PyMuPDF page extraction -> cache/pages/
     - E5 dense page embeddings -> cache/dense/
     - optional ColQwen page/query embeddings -> cache/visual/, cache/query_embs.pt
  -> Phase 2: python run.py
     - BM25 + dense E5 -> RRF -> adaptive page pruning
     - all-scanned PDFs use cached ColQwen MaxSim to choose VLM image pages
     - Qwen3.6-27B via vLLM or Transformers
  -> results.json, summary.json, <split>_submission.csv
```

`config.yaml` is the source of truth. Runtime overrides use OmegaConf dotlist syntax, for example `data.split=train retriever.max_pages=10`.

## Repo Structure

```text
LAVA2026/
  config.yaml
  run.py
  requirements.txt
  CONTEXT.md
  src/
    config.py
    evaluate.py
    utils/
      pdf_utils.py
      output_parser.py
      retriever.py
      visual_retriever.py
  scripts/job_lava.slurm
  lava-challenge-2026/      # dataset, gitignored
  cache/                    # preprocessing cache, gitignored
  runs/                     # experiment outputs, gitignored
```

## Setup

```bash
conda create -n vqa-jv python=3.11 -y
conda activate vqa-jv
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Main dependencies: `vllm`, `bitsandbytes`, `omegaconf`, `pymupdf`, `rank_bm25`, `sentence-transformers`, `faiss-cpu`, `fugashi[unidic-lite]`, `colpali-engine`.

## Run

```bash
# Smoke test.
python run.py --config config.yaml data.sample=1 model.qwen.path=/path/to/Qwen3.6-27B

# Train/test split.
python run.py --config config.yaml data.split=train model.qwen.path=/path/to/Qwen3.6-27B
python run.py --config config.yaml data.split=test  model.qwen.path=/path/to/Qwen3.6-27B

# Phase 1 cache build only.
python run.py --config config.yaml --preprocess --clear-cache data.split=test

# vLLM inference. Start vLLM first, or use the SLURM script.
python run.py --config config.yaml data.split=test model.qwen.backend=vllm

# Common tuning.
python run.py --config config.yaml retriever.max_pages=10 retriever.tokenizer=mecab
python run.py --config config.yaml retriever.dense_model=none
python run.py --config config.yaml visual.enabled=false vlm.enabled=false
```

Default outputs go to `runs/<split>_results.json`. Each run also writes `config_resolved.yaml`, `summary.json`, and `<split>_submission.csv` in the same directory.

## SLURM

```bash
cd /datastore/$USER/LAVA2026
sbatch scripts/job_lava.slurm

OVERRIDES="data.split=train data.sample=20" sbatch scripts/job_lava.slurm
OVERRIDES="retriever.max_pages=10 vlm.max_images_per_prompt=4" sbatch scripts/job_lava.slurm

tail -f runs/<JOB_ID>/job.out
tail -f runs/<JOB_ID>/job.err
tail -f runs/<JOB_ID>/vllm_server.log
```

The script assumes:

```text
PROJECT_DIR=/datastore/$USER/LAVA2026
PYTHON=/datastore/$USER/miniconda3/envs/vqa-jv/bin/python
VLLM=/datastore/$USER/miniconda3/envs/vqa-jv/bin/vllm
MODEL_DIR=/datastore/$USER/models/Qwen3.6-27B
COLQWEN_DIR=/datastore/$USER/models/colqwen2.5-v0.2
```

SLURM runs Phase 1 first, then starts vLLM with bitsandbytes quantization and image support, then runs Phase 2 against that server.

## Notes

- Transformers backend loads Qwen3.6-27B with 4-bit NF4.
- Production vLLM uses `--gpu-memory-utilization 0.50`, `--enforce-eager`, and `--limit-mm-per-prompt '{"image": 6}'`.
- `run.py` currently uses ColQwen only for visual cache and all-scanned VLM image selection, not for text-page RRF.
- `format_evidence_pages_for_csv()` defaults to `[1]` when evidence pages are empty.

## Submission Format

| Column | Format | Example |
| --- | --- | --- |
| `id` | matches `test.csv` | `q_0016` |
| `answer` | scalar: plain string; list: Python single-quote list | `43`, `['a', 'b']` |
| `evidence_page_number` | integer list string | `[1]`, `[1, 5, 8]` |
