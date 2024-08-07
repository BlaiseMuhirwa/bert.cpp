install:
	poetry install --no-root 
format:
	poetry run black src/python

train-bert:
	poetry run python src/python/main.py