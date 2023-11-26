FROM pytorch/pytorch:latest


# Install pip requirements

COPY requirements.txt .

RUN python -m pip install -r requirements.txt


WORKDIR /home/docker
COPY ./models/ /home/docker/models/
COPY ./utils/ /home/docker/utils/
COPY ./predict /home/docker/predict/

CMD python -m predict.predict_lstm
