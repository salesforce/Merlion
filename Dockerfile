FROM python:3.9-slim

# Install Java
RUN apt-get update \
  && apt-get install -y --no-install-recommends openjdk-11-jre-headless \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Install Merlion from source
# FIXME: install from pip once it's vetted
WORKDIR /workspace
COPY *.md .
COPY setup.py .
COPY merlion merlion
RUN python3 -m pip install "./[spark,prophet]"

# Copy Merlion pyspark apps
COPY pyspark_anomaly.py .
COPY pyspark_forecast.py .
