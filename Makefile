PY := python
VENV := .venv
PYBIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
MODEL := model.pth
ZIP := leaffliction.zip

.PHONY: all venv requirements clean fclean re lint help train predict test


all: venv requirements lint
		@echo 'To activate the virtual environment, run: source $(VENV)/bin/activate'

help:
		@echo "Available targets:"
		@echo "  all          - Set up venv, install requirements, and run lint"
		@echo "  venv         - Create virtual environment"
		@echo "  requirements - Install dependencies from requirements.txt"
		@echo "  lint         - Run flake8 linter on source code"
		@echo "  train        - Train the model with images/ dataset"
		@echo "  predict      - Run prediction (use: make predict IMAGE=path/to/image.jpg)"
		@echo "  test         - Test prediction accuracy on all categories"
		@echo "  extract      - Extract model.pth from leaffliction.zip"
		@echo "  clean        - Remove generated files and __pycache__"
		@echo "  fclean       - Remove venv and all generated files"
		@echo "  re           - Clean and rebuild everything"

lint:
		@echo "Running flake8..."
		@$(PYBIN) -m flake8 --exclude=$(VENV); rc=$$?; \
		if [ $$rc -eq 0 ]; then \
				echo "No issues"; \
		fi; \
		true
		@echo

venv:
		@echo "Checking for virtual environment..."
		@if [ ! -d "$(VENV)" ]; then \
				$(PY) -m venv $(VENV); \
				echo "Virtual environnement created in $(VENV)"; \
		else \
				echo "Virtual environment already exists"; \
		fi
		@echo

requirements: venv
		@echo "Installing dependencies..."
		$(PIP) install --upgrade pip
		$(PIP) install -r requirements.txt
		@echo

train: venv requirements
		@echo "Training model with images/ dataset..."
		@$(PYBIN) src/train.py --input images
		@echo "Training complete! Archive created: $(ZIP)"
		@echo

extract:
		@echo "Extracting model from archive..."
		@if [ -f "$(ZIP)" ]; then \
				unzip -j $(ZIP) $(MODEL) -d .; \
				echo "Model extracted: $(MODEL)"; \
		else \
				echo "Error: $(ZIP) not found. Run 'make train' first."; \
				exit 1; \
		fi
		@echo

predict: venv
		@if [ ! -f "$(MODEL)" ]; then \
				echo "Error: $(MODEL) not found. Run 'make extract' or 'make train' first."; \
				exit 1; \
		fi
		@if [ -z "$(IMAGE)" ]; then \
				echo "Running prediction on default sample image..."; \
				$(PYBIN) src/predict.py --model $(MODEL) --input "images/Apple_Black_rot/image (1).JPG"; \
		else \
				echo "Running prediction on $(IMAGE)..."; \
				$(PYBIN) src/predict.py --model $(MODEL) --input "$(IMAGE)"; \
		fi
		@echo

test: venv
		@echo "Testing prediction accuracy on all categories..."
		@if [ ! -f "$(MODEL)" ]; then \
				echo "Error: $(MODEL) not found. Run 'make extract' or 'make train' first."; \
				exit 1; \
		fi
		@for category in images/*/; do \
				if [ -d "$$category" ]; then \
						echo "Testing $$category..."; \
						$(PYBIN) src/predict.py --model $(MODEL) --input "$$category"; \
						echo ""; \
				fi; \
		done
		@echo "All tests complete!"

clean:
		@echo "Cleaning up..."
		-rm -f $(MODEL)
		-rm -f $(ZIP)
		-find . -type d -name __pycache__ -exec rm -rf {} +
		-find . -type f -name "*.pyc" -delete
		-find . -type f -name "*.pyo" -delete
		-find . -type f -name "*~" -delete
		@echo

fclean: clean
		@echo "Removing virtual environment..."
		-rm -rf $(VENV)
		@echo

re: fclean all