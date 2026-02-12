.PHONY: format lint test

format:
	ruff format src/
	ruff check --fix --select "I,RUF022" src/

lint:
	ruff check src/

test:
	pytest
