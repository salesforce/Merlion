FROM python:3.9-slim
WORKDIR /opt/Merlion
COPY *.md ./
COPY setup.py ./
COPY merlion merlion
RUN pip install gunicorn "./[dashboard]"
CMD gunicorn -b 0.0.0.0:80 merlion.dashboard.app:server
