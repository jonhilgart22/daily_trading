.PHONY: help install test lint build_inference build_training
.DEFAULT: help
PYTHON_VERSION=3.6.10

help:
	@echo "make install"
	@echo "       installs package dependencies"
	@echo "make test"
	@echo "       run tests"
	@echo "make lint"
	@echo "       format with black and isort, lint with flake8 and mypy"
	@echo "make build_inference"
	@echo "       build and push the inference image"
	@echo "make build_training"
	@echo "       build and push the training image"
	@exit 0

install:
	brew list pyenv || brew install pyenv
	pyenv install ${PYTHON_VERSION} -s
	pyenv local ${PYTHON_VERSION}
	poetry run pip install --upgrade pip
	poetry install --no-root

test: install
	poetry run python -m pytest

lint: install
	poetry run isort .
	poetry run black ml_smoking_status tests
	poetry run flake8 ml_smoking_status tests
	poetry run mypy ml_smoking_status tests

build_inference:
	./build_inference.sh --push

build_training:
	./build_training.sh --push
