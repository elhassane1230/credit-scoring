.PHONY: help install install-dev data eda train ablation all test lint app docker clean

PY := PYTHONPATH=src python3

help:
	@echo "Credit Scoring — tasks"
	@echo "  make install      Install runtime dependencies"
	@echo "  make data         Generate synthetic data + load into SQLite"
	@echo "  make eda          Run EDA (figures + solvency ranking)"
	@echo "  make train        Run the full ETL->train->select pipeline"
	@echo "  make ablation     Run the ablation studies"
	@echo "  make all          data + eda + train + ablation"
	@echo "  make test         Run the test suite"
	@echo "  make app          Launch the Flask app (http://localhost:5000)"
	@echo "  make docker       Build the Docker image"

install:
	pip install -r requirements.txt && pip install -e .

install-dev:
	pip install -r requirements-dev.txt && pip install -e .

data:
	$(PY) -c "from creditscore.config import get_config as g; from creditscore.data import generate, create_database; c=g(); create_database(generate(c.data), c.paths.db); print('DB ->', c.paths.db)"

eda:
	$(PY) scripts/run_eda.py

train:
	$(PY) scripts/run_pipeline.py

ablation:
	$(PY) scripts/run_ablation.py

all: train eda ablation

test:
	$(PY) -m pytest tests/ -q

lint:
	ruff check src tests scripts

app:
	$(PY) -m flask --app creditscore.app.flask_app run --port 5000

docker:
	docker build -t creditscore:latest .

clean:
	rm -rf data/*.db data/*.csv models/*.joblib models/*.json reports/*.json reports/figures/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
