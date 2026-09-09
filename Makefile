.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-dev store ingest app eval eval-ablation eval-judge lint fmt test cov clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev:  ## Install runtime + dev dependencies, editable package
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e .

store:  ## Extract the shipped vector store (data/chroma.tar.gz -> data/chroma/)
	$(PY) src/store.py

ingest:  ## Rebuild the vector store from the frozen corpus, then repack the tarball
	$(PY) src/ingest.py --reset

app:  ## Launch the Streamlit UI
	streamlit run src/app.py

eval:  ## Retrieval metrics (hit-rate@k, MRR) for the configured pipeline
	$(PY) src/evaluate.py

eval-ablation:  ## Full ablation matrix: dense / hybrid / +rerank
	$(PY) src/evaluate.py --ablation

eval-judge:  ## Also run LLM-as-judge answer faithfulness (slow, needs a generator)
	$(PY) src/evaluate.py --judge

lint:  ## Ruff lint
	ruff check src tests

fmt:  ## Ruff autofix + format
	ruff check --fix src tests && ruff format src tests

test:  ## Run the test suite
	pytest

cov:  ## Tests with coverage report
	pytest --cov=src --cov-report=term-missing

clean:  ## Remove caches
	find . -type d -name __pycache__ -exec rm -rf {} + ; rm -rf .pytest_cache .ruff_cache
