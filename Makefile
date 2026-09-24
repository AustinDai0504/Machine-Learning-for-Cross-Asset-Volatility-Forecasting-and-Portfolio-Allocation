.PHONY: install run offline report test lint
install:
	python -m pip install -r requirements-lock.txt
	python -m pip install -e . --no-deps
run:
	crossvol run
offline:
	crossvol run --offline
report:
	crossvol report
test:
	pytest -q
lint:
	ruff check src tests scripts
