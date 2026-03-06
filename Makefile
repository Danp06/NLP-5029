.PHONY: help install test lint format clean docker

help:
	@echo "Comandos disponibles:"
	@echo "  make install   — Instala en modo desarrollo"
	@echo "  make test      — Tests con cobertura"
	@echo "  make lint      — Verifica estilo (ruff + mypy)"
	@echo "  make format    — Formatea código (black)"
	@echo "  make clean     — Limpia archivos temporales"

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info

docker:
	docker build -t NLP-5029 .
