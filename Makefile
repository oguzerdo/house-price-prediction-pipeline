# GITHUB

commit:
	git commit -am "commit from make file"

push:
	git push origin main

pull:
	git pull origin main

fetch:
	git fetch origin main

reset:
	rm -f .git/index
	git reset

req:
	pip freeze > requirements.txt

compush: commit push

run:	
	python main.py --no-debug

run_debug:
	python main.py

run_test:
	pthon main.py --no-tuning

req:
	pip list --format=freeze > requirements.txt

install:
	pip install -r requirements.txt


models:
	python scripts/model_history.py

