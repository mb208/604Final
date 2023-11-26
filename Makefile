# Makefile 
.PHONY: build rerun download train predict clean setup

build: download train dockerbuild

rerun: clean build

download: 
	python source.loaddata.py

train:
	python train.lstmmodeltraining.py

predictions:
	docker run predict

dockerbuild:
	cd predict && docker build -t predict .
	docker tag predict:latest mgb208/predict 
	docker push mgb208/predict

setup:
	pip install -r requirements.txt

clean:
	rm -f data/*
	rm -f models/*