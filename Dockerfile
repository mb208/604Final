# FROM pytorch/pytorch:latest
FROM bitnami/pytorch:latest
USER root


# Install pip requirements

COPY requirements.txt .

RUN python -m pip install -r requirements.txt


WORKDIR /home/docker
COPY ./models/ /home/docker/models/
COPY ./data/ /home/docker/data/
COPY ./utils/ /home/docker/utils/
COPY ./source/ /home/docker/source/
COPY ./train/lstmmodeltraining.py /home/docker/train.py
COPY ./predict/predict_lstm.py /home/docker/predict.py
RUN touch __init__.py

ENTRYPOINT ["python", "-m"] 
