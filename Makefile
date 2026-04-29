.PHONY: help data pretrain sft dpo eval

help:
	@echo "TinyMathReason-1B Pipeline Commands"
	@echo "-----------------------------------"
	@echo "make data      - Runs the entire data processing pipeline (download -> shard)"
	@echo "make pretrain  - See docs/pretraining_setup.md for TPU instructions"
	@echo "make sft       - Prepares SFT data and starts SFT training (requires Thunder Compute / GPU)"
	@echo "make dpo       - Generates preferences and starts DPO training"
	@echo "make eval      - Runs the evaluation suite on the final model"

data:
	cd src/data/pipeline && $(MAKE) all

pretrain:
	@echo "Pretraining must be run on the TPU VM. Please refer to docs/pretraining_setup.md."

sft:
	cd src/sft && python prepare_sft_data.py
	cd src/sft && accelerate launch train_sft.py

dpo:
	cd src/dpo && python generate_preferences.py --model_path ../sft_output/final
	cd src/dpo && python train_dpo.py --model_path ../sft_output/final

eval:
	cd src/eval && python run_benchmarks.py --model_path ../dpo_output/final
	cd src/eval && python run_custom_eval.py --model_path ../dpo_output/final
	cd src/eval && python generate_comparison.py
