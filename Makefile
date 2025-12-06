# --- Project Configuration ---
DATASET ?= liepa2
N_SPEAKERS_PER_GENDER ?= 15
EXECUTOR := uv run
CONFIG_FILE := configs/config.json

# --- Paths ---
# Raw data input paths
RAW_LIEPA2_DIR := data/raw/liepa2
RAW_LIEPA2_CHECK_FILE := $(RAW_LIEPA2_DIR)/train-00000-of-00130.parquet

# Processed data output paths
PROCESSED_LIEPA2_DIR := data/processed/tts_dataset_liepa2_$(shell echo $$(($(N_SPEAKERS_PER_GENDER) * 2)))spk
PROCESSED_LIEPA2_CHECK_FILE := $(PROCESSED_LIEPA2_DIR)/metadata.csv

# Embedding output paths
EMBEDDINGS_LIEPA2_DIR := $(PROCESSED_LIEPA2_DIR)
EMBEDDINGS_LIEPA2_CHECK_FILE := $(EMBEDDINGS_LIEPA2_DIR)/speakers.pth

# Training output path (defined inside config.json, but used for cleanup)
TRAIN_OUTPUT_DIR := training_output

# --- Targets ---
.PHONY: all install data train clean format help

# Default target: runs the full pipeline
all: data train ## Run the full pipeline: install dependencies, process data, and train the model.

# 1. Setup Environment
install: ## Install required Python packages using uv.
	uv sync
	@echo "\nDependencies installed."

# Liepa-2 dataset processing
$(PROCESSED_LIEPA2_CHECK_FILE): $(RAW_LIEPA2_CHECK_FILE) python/preprocess_liepa2.py
	@echo "Processing Liepa-2 data ($(shell echo $$(($(N_SPEAKERS_PER_GENDER) * 2))) speakers)..."
	$(EXECUTOR) python python/preprocess_liepa2.py --input_path $(RAW_LIEPA2_DIR) --output_path $(PROCESSED_LIEPA2_DIR) --n_speakers_per_gender $(N_SPEAKERS_PER_GENDER)
	@echo "\nLiepa-2 data preprocessing complete. Output at: $(PROCESSED_LIEPA2_DIR)"

# Data processing targets
data-liepa2: $(PROCESSED_LIEPA2_CHECK_FILE) ## Preprocess the Liepa-2 dataset.
data: data-liepa2 ## Preprocess the Liepa-2 dataset for training.

# 2.5. Compute Speaker Embeddings
# Liepa-2 speaker embeddings computation
$(EMBEDDINGS_LIEPA2_CHECK_FILE): $(PROCESSED_LIEPA2_CHECK_FILE) python/compute_embeddings.py
	@echo "Computing speaker embeddings for the Liepa-2 dataset..."
	$(EXECUTOR) python python/compute_embeddings.py \
	  --dataset_path $(PROCESSED_LIEPA2_DIR) \
	  --output_path $(EMBEDDINGS_LIEPA2_CHECK_FILE)
	@echo "\nSpeaker embeddings computed. Output at: $(EMBEDDINGS_LIEPA2_CHECK_FILE)"

# Embedding computation targets
compute-embeddings-liepa2: $(EMBEDDINGS_LIEPA2_CHECK_FILE) ## Compute speaker embeddings for the Liepa-2 dataset.
compute-embeddings: compute-embeddings-liepa2 ## Compute speaker embeddings for the dataset.

# 3. Train Model
# This target depends on the processed data, embeddings, and the config file.
train: compute-embeddings-liepa2 $(CONFIG_FILE) ## Start or resume the TTS model training with Liepa-2.
	@echo "Launching TTS training with $(shell echo $$(($(N_SPEAKERS_PER_GENDER) * 2))) speakers using config '$(CONFIG_FILE)'..."
	@echo "Dataset path: $(PROCESSED_LIEPA2_DIR)"
	@echo "Speaker embeddings: $(EMBEDDINGS_LIEPA2_CHECK_FILE)"
	@echo "Number of speakers: $(shell echo $$(($(N_SPEAKERS_PER_GENDER) * 2)))"
	$(EXECUTOR) python python/train.py \
	  --config_path $(CONFIG_FILE) \
	  --coqpit.datasets.0.path=$(PROCESSED_LIEPA2_DIR)/ \
	  --coqpit.speakers_file=$(PROCESSED_LIEPA2_DIR)/speakers.json \
	  --coqpit.d_vector_file=$(EMBEDDINGS_LIEPA2_CHECK_FILE) \
	  --coqpit.num_speakers=$(shell echo $$(($(N_SPEAKERS_PER_GENDER) * 2)))
	@echo "\nTraining finished. Check results in '$(TRAIN_OUTPUT_DIR)'."

# 5. Format Code
format: ## Format Python code using ruff.
	@echo "Formatting Python code with ruff..."
	$(EXECUTOR) ruff format .
	@echo "\nCode formatting complete."

# 6. Cleanup
clean: ## Remove all generated files (processed data, training output, cache).
	@echo "Cleaning up project directory..."
	rm -rf data/processed/*
	rm -rf $(TRAIN_OUTPUT_DIR)
	rm -rf .cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "\nCleanup complete."

clean-embeddings: ## Remove only computed embeddings.
	@echo "Cleaning up speaker embeddings..."
	rm -f $(EMBEDDINGS_LIEPA2_CHECK_FILE)
	@echo "\nEmbeddings cleanup complete."

# 7. Inference
inference: ## Run inference on test samples.
	@echo "Running inference on test samples..."
	$(EXECUTOR) python python/run_inference.py \
	  --use_gpu \
	  --model_dir $(TRAIN_OUTPUT_DIR)/Tacotron2-DCA-November-06-2025_10+04PM-6805a9d/ \
	  --vocoder_path "/home/aleks/.local/share/tts/vocoder_models--en--ljspeech--multiband-melgan/model.pth" \
	  --speakers VP382,VP460
	@echo "\nInference complete. Check 'inference_output/' for generated audio files."


# --- Helper for Missing Data ---
# This rule provides instructions if the raw data is not found.
$(RAW_LIEPA2_CHECK_FILE):
	@echo "------------------------------------------------------------------"
	@echo "ACTION REQUIRED: Liepa-2 raw data not found!"
	@echo "Please download the Liepa-2 dataset and place parquet files in:"
	@echo "    $(RAW_LIEPA2_DIR)/"
	@echo "Expected file: $(RAW_LIEPA2_CHECK_FILE)"
	@echo "------------------------------------------------------------------"
	@exit 1
