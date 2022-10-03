ARG spark_uid=185
FROM gcr.io/spark-operator/spark-py:v3.1.1

# Change to root user for installation steps
USER 0

# Install pyarrow (for spark-sql) and Merlion; get pyspark & py4j from the PYTHONPATH
ENV PYTHONPATH="${SPARK_HOME}/python/lib/pyspark.zip:${SPARK_HOME}/python/lib/py4j-0.10.9-src.zip:${PYTHONPATH}"
COPY *.md ./
COPY setup.py ./
COPY merlion merlion
RUN pip install pyarrow "./"

# Copy Merlion pyspark apps
COPY apps /opt/spark/apps
RUN chmod g+w /opt/spark/apps
USER ${spark_uid}
COPY emissions.csv emissions.csv
COPY emissions.json emissions.json