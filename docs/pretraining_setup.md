# Pretraining Setup (GCP TPU)

This phase utilizes Google Cloud TPUs for extremely fast and cost-effective pretraining.

## Hardware Strategy
We utilize two spot TPU clusters:
1. **`v5litepod-64` (Main Training):** 64 chips of v5e. This is our primary workhorse. MaxText scales perfectly here, yielding ~200k+ tokens/sec.
2. **`v4-32` (Auxiliary / Evals):** 32 chips of v4. Used to continuously pull checkpoints from GCS and run `lm-eval` to watch performance curves in real-time without pausing the main cluster.

## 1. Setup TPU Environment

SSH into your TPU VM and install MaxText:

```bash
git clone https://github.com/google/maxtext.git
cd maxtext
bash setup.sh
```

## 2. Configure MaxText
Copy your configuration file (`src/train/maxtext_config.yml`) into the `maxtext/configs/` directory. Ensure `dataset_path` points to your GCS bucket.

## 3. Launching the Run on `v5litepod-64`

Run a quick 1000-step smoke test first:
```bash
python MaxText/train.py \
    MaxText/configs/tinymath_1b.yml \
    run_name=tinymath_smoke_test \
    steps=1000
```

Once verified, launch the main run:
```bash
python MaxText/train.py \
    MaxText/configs/tinymath_1b.yml \
    run_name=tinymath_main_run \
    steps=300000
```

## 4. Handling Preemptions
Since these are Spot VMs, they will be preempted. Run the included preemption handler daemon on the background:
```bash
nohup python src/train/preemption_handler.py \
  --project YOUR_PROJECT_ID \
  --zone YOUR_ZONE \
  --tpu_name YOUR_TPU_NAME \
  --script_path "python MaxText/train.py MaxText/configs/tinymath_1b.yml run_name=tinymath_main_run" &
```

## 5. Auxiliary Evals on `v4-32`
While the main run happens, SSH into your `v4-32` cluster. You can run `src/eval/run_benchmarks.py` directly by passing it the GCS checkpoint paths (once converted to HuggingFace format using the `convert_checkpoint.py` script).
