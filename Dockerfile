FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

RUN apt-get update && apt-get install  -y graphviz

# Copy the uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

WORKDIR /tf

ENV UV_SYSTEM_PYTHON=1 \
    TF_CPP_MIN_LOG_LEVEL=2\
    PYTHONPATH=/tf,/tf/src

COPY requirements.txt .

RUN uv pip install --no-cache-dir -r requirements.txt

COPY src ./src
