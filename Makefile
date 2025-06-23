# Tsukuyomi TTS Makefile

.PHONY: help install install-dev test test-fast test-gpu lint format clean build docs

help:
	@echo "Available commands:"
	@echo "  make install      Install package and dependencies"
	@echo "  make install-dev  Install with development dependencies"
	@echo "  make test         Run all tests"
	@echo "  make test-fast    Run tests excluding slow tests"
	@echo "  make test-gpu     Run GPU-specific tests"
	@echo "  make lint         Run linting checks"
	@echo "  make format       Format code"
	@echo "  make clean        Clean build artifacts"
	@echo "  make build        Build distribution packages"
	@echo "  make docs         Build documentation"

# Installation
install:
	uv venv
	uv pip install -e .

install-dev:
	uv venv
	uv pip install -e .[dev,test,h100]

install-h100:
	uv venv
	uv pip install -e .[h100]
	bash scripts/setup_h100_training.sh

# Testing
test:
	uv run pytest -v --cov=src --cov-report=html --cov-report=term

test-fast:
	uv run pytest -v -m "not slow and not gpu" --cov=src

test-gpu:
	uv run pytest -v -m "gpu" --cov=src

test-integration:
	uv run pytest tests/test_integration.py -v

test-performance:
	uv run pytest tests/test_performance.py -v -m "benchmark"

# Code quality
lint:
	uv run ruff check src tests
	uv run black --check src tests
	uv run isort --check-only src tests
	uv run mypy src

format:
	uv run black src tests
	uv run isort src tests
	uv run ruff check --fix src tests

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building
build: clean
	uv build

# Documentation
docs:
	@echo "Documentation building not yet configured"
	@echo "See docs/ directory for documentation"

# Development helpers
dev-setup: install-dev
	pre-commit install

run-example:
	uv run python scripts/test_synthesis.py

benchmark:
	uv run python scripts/test_synthesis.py --benchmark

# Docker (future)
docker-build:
	@echo "Docker support coming soon"

# Training
train-setup:
	bash scripts/setup_h100_training.sh

train-monitor:
	uv run python scripts/monitor_training.py