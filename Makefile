# ==============================================================================
# Makefile for the End-to-End TTS Thesis Project
# ==============================================================================

# --- Project Configuration ---
# Use `make DATASET=liepa2-multispeaker` or `make DATASET=liepa2` to choose dataset
DATASET ?= liepa2
N_SPEAKERS ?= 20
EXECUTOR := uv run
CONFIG_FILE := configs/config.json
CONFIG_FILE_MULTISPEAKER := configs/config_multispeaker.json

# --- Paths ---
# Raw data input paths
RAW_LIEPA2_DIR := data/raw/liepa2
RAW_LIEPA2_CHECK_FILE := $(RAW_LIEPA2_DIR)/train-00000-of-00130.parquet

# Processed data output paths
PROCESSED_LIEPA2_DIR := data/processed/tts_dataset_liepa2
PROCESSED_LIEPA2_CHECK_FILE := $(PROCESSED_LIEPA2_DIR)/metadata.csv
PROCESSED_LIEPA2_MULTISPEAKER_DIR := data/processed/tts_dataset_liepa2_multispeaker
PROCESSED_LIEPA2_MULTISPEAKER_CHECK_FILE := $(PROCESSED_LIEPA2_MULTISPEAKER_DIR)/metadata.csv

# Training output path (defined inside config.json, but used for cleanup)
TRAIN_OUTPUT_DIR := training_output

# --- Targets ---
.PHONY: all install data train clean format help

# Default target: runs the full pipeline
all: data train ## Run the full pipeline: install dependencies, process data, and train the model.

help: ## Show this help message.
	@echo "Usage: make [target] [VARIABLE=value]"
	@echo ""
	@echo "Examples:"
	@echo "  make data DATASET=liepa2             # Process only single-speaker Liepa-2"
	@echo "  make data DATASET=liepa2-multispeaker # Process multispeaker Liepa-2"
	@echo "  make train                           # Train with single-speaker Liepa-2"
	@echo "  make train-multispeaker              # Train with multispeaker Liepa-2"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# 1. Setup Environment
install: ## Install required Python packages using uv.
	uv sync
	@echo "\nDependencies installed."

# 2. Process Data
# Liepa-2 dataset processing (single-speaker)
$(PROCESSED_LIEPA2_CHECK_FILE): $(RAW_LIEPA2_CHECK_FILE) python/preprocess_liepa2.py
	@echo "Processing Liepa-2 data (single-speaker)..."
	$(EXECUTOR) python python/preprocess_liepa2.py --liepa_path $(RAW_LIEPA2_DIR) --output_path $(PROCESSED_LIEPA2_DIR)
	@echo "\nLiepa-2 single-speaker data preprocessing complete. Output at: $(PROCESSED_LIEPA2_DIR)"

# Liepa-2 dataset processing (multispeaker)
$(PROCESSED_LIEPA2_MULTISPEAKER_CHECK_FILE): $(RAW_LIEPA2_CHECK_FILE) python/preprocess_liepa2_multispeaker.py
	@echo "Processing Liepa-2 data (multispeaker with $(N_SPEAKERS) speakers)..."
	$(EXECUTOR) python python/preprocess_liepa2_multispeaker.py --liepa_path $(RAW_LIEPA2_DIR) --output_path $(PROCESSED_LIEPA2_MULTISPEAKER_DIR) --n_speakers $(N_SPEAKERS)
	@echo "\nLiepa-2 multispeaker data preprocessing complete. Output at: $(PROCESSED_LIEPA2_MULTISPEAKER_DIR)"

# Data processing targets
data-liepa2: $(PROCESSED_LIEPA2_CHECK_FILE) ## Preprocess the single-speaker Liepa-2 dataset.
data-liepa2-multispeaker: $(PROCESSED_LIEPA2_MULTISPEAKER_CHECK_FILE) ## Preprocess the multispeaker Liepa-2 dataset.

ifeq ($(DATASET),liepa2-multispeaker)
data: data-liepa2-multispeaker ## Preprocess the multispeaker Liepa-2 dataset for training.
else
data: data-liepa2 ## Preprocess the single-speaker Liepa-2 dataset for training.
endif

# 3. Train Model
# This target depends on the processed data and the config file.
train: data $(CONFIG_FILE) ## Start or resume the TTS model training with single-speaker Liepa-2.
	@echo "Launching TTS training with config '$(CONFIG_FILE)'..."
	$(EXECUTOR) python -m TTS.bin.train_tts --config_path $(CONFIG_FILE)
	@echo "\nTraining finished. Check results in '$(TRAIN_OUTPUT_DIR)'."

# 4. Train Multispeaker Model
# This target depends on the processed data and the config file.
train-multispeaker: data-liepa2-multispeaker $(CONFIG_FILE_MULTISPEAKER) ## Start or resume the TTS model training with multispeaker Liepa-2.
	@echo "Launching TTS training with config '$(CONFIG_FILE_MULTISPEAKER)'..."
	$(EXECUTOR) python python/train.py --config_path $(CONFIG_FILE_MULTISPEAKER)
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
