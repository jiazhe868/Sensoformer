# Makefile for Sensoformer
# Includes: Compilation of Fortran extensions, Installation, Testing, and Cleaning

# --- Configuration ---
PYTHON = python
PIP = pip
FC = gfortran
# Flags: Shared library, Position Independent Code, Optimization Level 3
FFLAGS = -shared -fPIC -O3

# --- Paths ---
EXT_DIR = src/sensoformer/ext
LIB_NAME = mtdcmp.so
SRC_NAME = mtdcmp.f

# --- Targets ---

.PHONY: all build clean test install help weights demo lint

help:
	@echo "Sensoformer Engineering Makefile"
	@echo "=============================="
	@echo "make build    : Compile Fortran extensions"
	@echo "make install  : Install package in editable mode"
	@echo "make test     : Run unit tests with pytest"
	@echo "make clean    : Remove build artifacts and compiled libraries"
	@echo "make weights  : Download pretrained weights from the Hugging Face Hub"
	@echo "make demo     : Download weights + SoCal data, then run inference"
	@echo "make all      : Clean, Build, Install, and Test"

all: clean build install test

# 1. Compile Fortran Extension
build:
	@echo "--> Compiling Fortran extension..."
	@if [ -f "$(EXT_DIR)/$(SRC_NAME)" ]; then \
		$(FC) $(FFLAGS) $(EXT_DIR)/$(SRC_NAME) -o $(EXT_DIR)/$(LIB_NAME); \
		echo "--> Compilation success: $(EXT_DIR)/$(LIB_NAME)"; \
	else \
		echo "--> Error: Source file $(EXT_DIR)/$(SRC_NAME) not found!"; \
		exit 1; \
	fi

# 2. Install Package
install:
	@echo "--> Installing sensoformer in editable mode..."
	$(PIP) install -e .[dev]

# 3. Run Tests
test:
	@echo "--> Running Unit Tests..."
	pytest tests/ -v

# 4. Clean Up
clean:
	@echo "--> Cleaning up..."
	rm -f $(EXT_DIR)/$(LIB_NAME)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/
# 5. Fetch pretrained weights (~8 MB each)
weights:
	@echo "--> Downloading pretrained weights..."
	$(PYTHON) scripts/download_assets.py --weights

# 6. End-to-end demo: weights + the 0.26 GB real catalog + inference
demo:
	@echo "--> Downloading assets (this pulls ~0.26 GB of data)..."
	$(PYTHON) scripts/download_assets.py --weights --datasets socal-real
	@echo "--> Running inference on 200 events..."
	$(PYTHON) scripts/predict.py --input socal-real --out-dir results/demo \
		--limit 200 --figures
	@echo "--> See results/demo/{predictions.csv,metrics.json}"
