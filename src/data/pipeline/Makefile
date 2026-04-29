.PHONY: all download clean mix pack shard upload

DATA_DIR ?= /data
GCS_BUCKET ?= gs://tinymath-bucket/pretraining-data/
TOKENIZER_PATH ?= ./tokenizer

all: download clean mix pack shard upload

download:
	python a_download_datasets.py --output_dir $(DATA_DIR)/raw

clean:
	python b_clean_and_filter.py --input_dir $(DATA_DIR)/raw --output_dir $(DATA_DIR)/clean

mix:
	python c_mix_datasets.py --input_dir $(DATA_DIR)/clean --output_dir $(DATA_DIR)/mixed

pack:
	python d_tokenize_and_pack.py --input_dir $(DATA_DIR)/mixed --output_dir $(DATA_DIR)/packed --tokenizer_path $(TOKENIZER_PATH) --seq_len 4096

shard:
	python e_create_shards.py --input_dir $(DATA_DIR)/packed --output_dir $(DATA_DIR)/shards

upload:
	python f_upload_to_gcs.py --input_dir $(DATA_DIR)/shards --gcs_bucket $(GCS_BUCKET)
