FROM python:3.9-slim
WORKDIR /opt/Merlion
# Install Java
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*
# Install Merlion from source & set up a gunicorn server
COPY *.md ./
COPY setup.py ./
COPY merlion merlion
RUN pip install gunicorn "./[dashboard]"

RUN pip install 'dash_bootstrap_components<=1.7.1' --force-reinstall

RUN pip install 'dash<=2.18.0' --force-reinstall

CMD gunicorn -b 0.0.0.0:80 merlion.dashboard.server:server
