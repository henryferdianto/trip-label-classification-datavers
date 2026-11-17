PY ?= python3
VENV ?= .venv

.PHONY: setup sample run test clean

setup:
	$(PY) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

sample:
	$(PY) -m src.sample_data

run:
	$(PY) -m src.run_demo

test:
	$(PY) -m pytest -q

clean:
	rm -rf $(VENV) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
