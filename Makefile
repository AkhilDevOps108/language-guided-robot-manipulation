PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: venv install collect validate train evaluate report serve test mlflow-ui format-clean

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install:
	$(PIP) install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

collect:
	$(PYTHON) -m gripground.data.collect_demonstrations --output-dir artifacts/data --episodes 90 --max-steps 60

validate:
	$(PYTHON) -m gripground.data.validate_dataset --dataset-dir artifacts/data

train:
	$(PYTHON) -m gripground.training.train_policy --dataset-dir artifacts/data --output-dir artifacts/checkpoints --epochs 50 --batch-size 64

evaluate:
	$(PYTHON) -m gripground.training.evaluate_policy --dataset-dir artifacts/data --checkpoint artifacts/checkpoints/best.pt --reports-dir reports --episodes 30

report:
	$(PYTHON) -m gripground.evaluation.generate_report --reports-dir reports

serve:
	GRIPGROUND_MODEL_PATH=artifacts/checkpoints/best.pt $(PYTHON) -m uvicorn gripground.serving.app:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest -q

mlflow-ui:
	$(PYTHON) -m mlflow ui --backend-store-uri file:./artifacts/mlruns --host 127.0.0.1 --port 5000

format-clean:
	rm -rf .pytest_cache __pycache__ */__pycache__
