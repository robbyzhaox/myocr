.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch myocr/ docs/source/ docs/build/

.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v tests/

.PHONY : format-and-test
format-and-test : 
	isort .
	black .
	ruff check .
	mypy .
	CUDA_VISIBLE_DEVICES='' pytest -v tests/

.PHONY : build
build :
	rm -rf *.egg-info/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	python -m build
