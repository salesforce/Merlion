FROM python:3.9-slim
WORKDIR /opt/Merlion
# Install Java
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update && \
    apt-get upgrade && \
    apt-get install -y --no-install-recommends openjdk-11-jre-headless && \
    rm -rf /var/lib/apt/lists/*
# Install Merlion from source & set up a gunicorn server
COPY *.md ./
COPY setup.py ./
COPY merlion merlion
RUN pip install gunicorn "./[dashboard]"
CMD gunicorn -b 0.0.0.0:80 merlion.dashboard.server:server
