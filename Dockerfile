ARG spark_uid=185
FROM apache/spark-py:v3.2.1

# Change to root user for installation steps
USER 0

# Uninstall existing python and replace it with miniconda.
# This is to get the right version of Python in Debian, since Prophet doesn't play nice with Python 3.9+.
# FIXME: maybe optimize the size? this image is currently 3.2GB.
RUN apt-get update && \
    apt-get remove -y python3 python3-pip && \
    apt-get install -y --no-install-recommends curl && \
    apt-get autoremove -yqq --purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -v -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    # Install prophet while we're at it, since this is easier to conda install than pip install
    /opt/conda/bin/conda install -y prophet && \
    /opt/conda/bin/conda clean -ya
ENV PATH="/opt/conda/bin:${SPARK_HOME}/bin:${PATH}"

# Install (for spark-sql) and Merlion; get pyspark & py4j from the PYTHONPATH
ENV PYTHONPATH="${SPARK_HOME}/python/lib/pyspark.zip:${SPARK_HOME}/python/lib/py4j-0.10.9.3-src.zip:${PYTHONPATH}"
COPY *.md ./
COPY setup.py ./
COPY merlion merlion
RUN pip install pyarrow "./[prophet]" && pip uninstall -y py4j

# Copy Merlion pyspark apps
COPY spark /opt/spark/apps
USER ${spark_uid}
