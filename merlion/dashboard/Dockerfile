FROM python:3.8-slim
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
RUN pip install gunicorn
COPY . ./
CMD gunicorn -b 0.0.0.0:80 app:server
