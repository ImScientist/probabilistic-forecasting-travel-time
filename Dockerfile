FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

RUN apt-get update && apt-get install graphviz -y

WORKDIR /tf

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONPATH=/tf,/tf/src
