ARG spark_uid=185
ARG PYTHON_VERSION=3.8
ARG SPARK_VERSION=3.2.1
FROM apache/spark-py:v${SPARK_VERSION}

# Change to root user for installation steps
USER 0

# Uninstall existing python and replace it with miniconda.
# This is to get the right version of Python in Debian, since Prophet doesn't play nice with Python 3.9+.
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
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build prophet && \
    /opt/conda/bin/conda clean -ya
ENV PATH="/opt/conda/bin:${SPARK_HOME}/bin:${PATH}"

# Install appropriate version of pyspark & then install Merlion from source
# We get spark version programmatically because the ARG/ENV gets overwritten for SPARK_VERSION
RUN pip install pyspark[sql]==$(spark-submit --version 2>&1 | grep -Eo 'version ([0-9]{1,}\.)+[0-9]{1,}' -m 1 | sed -e 's/version //')
COPY *.md ./
COPY setup.py ./
COPY merlion merlion
RUN pip install "./[spark,prophet]"

# Copy Merlion pyspark apps
COPY spark /opt/spark/apps
USER ${spark_uid}