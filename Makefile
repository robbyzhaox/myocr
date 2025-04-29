.PHONY : docs
docs :
	cd documentation && rm -rf site && mkdocs build

.PHONY : run-format
run-format :
	isort .
	black .
	ruff check . --fix
	mypy .

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v .

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build
