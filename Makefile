.PHONY: setup benchmark train dashboard test reset

setup:
	pip install -e ".[dev]"
	python -m deltaloop.storage.init_db
	ollama pull llama3.1:8b
	ollama pull mistral:7b

benchmark:
	python -m deltaloop.benchmark.runner

train:
	python -m deltaloop.training.dpo_trainer --manual

dashboard:
	mlflow ui --port 5000 &
	cd frontend && npm run dev

test:
	pytest tests/ -v --asyncio-mode=auto

reset:
	rm -f deltaloop.db
	rm -rf adapters/
	rm -rf mlruns/
