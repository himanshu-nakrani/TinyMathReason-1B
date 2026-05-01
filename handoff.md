Here is the complete, high-fidelity handoff document containing every critical detail, architectural decision, and pipeline nuance we covered. 

You can copy and paste this entire block into your next chat to instantly bring the AI up to speed on exactly where the project stands.

***

# TinyMathReason-1B: Project Handoff & State Summary

## 1. Project Objective & Architecture
**Goal:** Train a 1.1 Billion parameter Large Language Model (`TinyMathReason-1B`) from scratch, specializing in mathematical logic and algorithmic reasoning using a 2-stage curriculum SFT approach to elicit `<think>` reasoning traces (similar to DeepSeek-R1-Zero).
- **Architecture:** Standard Decoder-only Transformer (Llama-style architecture, ~1.1B parameters).
- **Tokenizer:** Custom BPE Tokenizer. **CRITICAL NOTE:** The tokenizer configuration `train_tokenizer.py` was updated to explicitly include `<think>` and `</think>` as special tokens to support native reasoning trace generation during post-training.
- **SFT Strategy:** We pivoted to a 2-Stage Curriculum SFT:
  - **Stage 1:** Conversational instruction-following (teaching the model to chat without Chain-of-Thought).
  - **Stage 2:** Reasoning trace fine-tuning. **Important:** Requires calling `--resize_token_embeddings` in the training script to fully integrate the `<think>` tags into the model's vocabulary.

## 2. Phase 1: Pretraining Data Pipeline (COMPLETED)
We successfully engineered and executed a massive data pipeline across two Vultr Bare Metal servers (Node A and Node B). Both servers have now been completely destroyed to halt billing.

**Total Pretraining Corpus Size:** ~57 Billion Tokens. 
*(According to Chinchilla scaling laws, 1.1B parameters require ~22B tokens for full saturation. With 57B tokens heavily weighted toward math and code, the model has more than double the necessary data for mastery).*

### Dataset Breakdown & GCS Storage Structure
All data is stored in the Google Cloud bucket: `gs://tinymath-reason-data-himanshu/pretraining-data/` in highly compressed `jsonl.zst` format perfectly structured for the MaxText TPU dataloader. Because shards share identical filenames (`shard_000000.jsonl.zst`), they were safely isolated into subdirectories to prevent overwriting.

1. **FineWeb-Edu (~10 Billion Tokens)**
   - **Shards:** 363 shards
   - **Location:** `gs://tinymath-reason-data-himanshu/pretraining-data/` (Root folder)
   - **Execution:** Processed entirely on Node A.
2. **GAIR/MathPile (~9.5 Billion Tokens)**
   - **Shards:** 225 shards
   - **Location:** `gs://tinymath-reason-data-himanshu/pretraining-data/mathpile/`
   - **Execution:** Processed on Node A. We swapped to MathPile because the original `EleutherAI/proof-pile-2` dataset was completely broken upstream on Hugging Face. MathPile is a strictly cleaner, more modern alternative.
3. **OpenWebMath + Stack-Edu (~37.7 Billion Tokens)**
   - **Shards:** 1,041 shards
   - **Location:** `gs://tinymath-reason-data-himanshu/pretraining-data/math-and-code/`
   - **Execution:** Processed entirely on Node B. Generated from ~45.4 Million individual math/code web documents.

## 3. Engineering Fixes & Pipeline Gotchas
*If the pipeline ever needs to be run again, be aware of the following fixes implemented in the repository:*
- **Hugging Face Dataset Scripts:** HF recently deprecated `trust_remote_code=True` for dataset scripts. For legacy datasets, we downgraded to `datasets==2.21.0`.
- **Zstandard Streaming Bug:** Python's `zstandard` crashes on HTTP range streaming. For `.zst` datasets, we set `streaming=False` to download to disk first. For `.gz` datasets (like MathPile), `streaming=True` safely bypasses Arrow schema errors.
- **OOM (Out-of-Memory) Kills:** When processing huge chunks, the `multiprocessing` workers crashed Linux due to RAM exhaustion. 
  - **Fix:** We hard-capped `num_cores = 14` in both `d_tokenize_and_pack.py` and `e_create_shards.py`. 
  - **Fix:** We added strict fault-tolerance and skip/resume logic (`if out_file.exists(): return True`) to the pipeline so if a crash occurs, it skips fully written shards and gracefully drops corrupted partial files.

## 4. Next Steps: Phase 2 (Pretraining)
The data pipeline is fully complete and all code changes were pushed to the `main` branch. 

**Immediate Objectives for Phase 2:**
1. Provision a Google Cloud `v5litepod-64` (TPU v5e cluster) using Google Kubernetes Engine (GKE) or directly via Queued Resources.
2. Set up the `MaxText` pretraining repository.
3. Configure the MaxText dataloader to read from the multiple subdirectories in `gs://tinymath-reason-data-himanshu/pretraining-data/`.
4. Execute a quick "Smoke Test" pretraining run (100 steps) to ensure the TPUs are correctly streaming the `.jsonl.zst` shards from the bucket without bottlenecks.
5. Launch the full Pretraining Run.

*** 

*End of Handoff Document*