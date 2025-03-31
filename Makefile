.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch myocr/ docs/source/ docs/build/

.PHONY : run-format
run-format :
	isort .
	black .
	ruff check .
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
