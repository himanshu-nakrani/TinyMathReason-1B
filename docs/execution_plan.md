# Execution Plan (Accelerated Resources)

| Phase | Tasks to Complete | Resource Used |
| :--- | :--- | :--- |
| **Phase 1: Setup & Data** | Setup Repo, Train Tokenizer. Spin up two massive CPU nodes. Process FineWeb on Node A, OpenWebMath/Proof/Stack on Node B. Pack & upload to GCS. | Local / Vultr (2x `c2-standard-30`) |
| **Phase 2: Pretraining** | Use the `v4-32` to train the 1.12B model and run intermediate evals via `lm-eval`. | GCP TPU `v4-32` |
| **Phase 3: SFT** | Convert Orbax to Safetensors. Stage 1: SFT on conversational prior. Stage 2: SFT on `<think>` reasoning traces. | AMD MI300X (192GB VRAM) |
| **Phase 4: DPO/GRPO** | Generate candidate solutions using serverless vLLM. Run DPO/GRPO training. | Modal + AMD MI300X |
| **Phase 5: Evaluation** | Run full benchmark suite, custom evals, and write final report. | Lightning AI + Thunder Compute |

## Critical Milestones & Deadlines
- **May 5th:** Vultr credits expire. All 300B tokens MUST be processed and uploaded to GCS.
- **May 15th:** SFT phase begins.
- **May 22nd:** TPU spot clusters expire. Pretraining must be completely finished and converted to HuggingFace format.
