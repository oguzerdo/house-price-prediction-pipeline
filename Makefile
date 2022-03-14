run:
	python main.py

req:
	pip list --format=freeze > requirements.txt

install:
	pip install -r requirements.txt



