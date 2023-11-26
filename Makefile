# Makefile 
.PHONY: build rerun download train predict clean

build: download train dockerbuild

rerun: clean build

download: 
	python source.loaddata.py
train:
	python train.lstmmodeltraining.py

predictions:
	docker run predict

dockerbuild:
	docker build -t predict .

clean:
	rm -f data/*
	rm -f models/*