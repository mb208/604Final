FROM pytorch/pytorch:latest


# Install pip requirements

COPY requirements.txt .

RUN python -m pip install -r requirements.txt


WORKDIR /home/docker
COPY ./models/ /home/docker/models/
COPY ./data/ /home/docker/data/
COPY ./utils/ /home/docker/utils/
COPY ./train/lstmmodeltraining.py /home/docker/train.py
COPY ./predict/predict_lstm.py /home/docker/predict.py
RUN touch __init__.py

ENTRYPOINT ["python", "-m"] 
