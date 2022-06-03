#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Utils for creating a ``pyspark.sql.SparkSession``.
"""
from typing import Dict

try:
    from pyspark import SparkConf
    from pyspark.sql import SparkSession
except ImportError as e:
    err = (
        "Try installing Merlion with optional dependencies using `pip install salesforce-merlion[spark]` or "
        "`pip install `salesforce-merlion[all]`"
    )
    raise ImportError(str(e) + ". " + err)


def create_session(name, **kwargs) -> SparkSession:
    """
    Creates a SparkSession with the specified app name and any additional options the user specifies.
    """
    # Create spark config with PyArrow enabled & other arguments specified
    conf = SparkConf()
    conf.setAppName(name)
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    for k, v in kwargs.items():
        conf.set(k, v)

    # Create spark session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark


def enable_aws_kwargs(credentials_provider=None) -> Dict[str, str]:
    """
    Returns a dict of config params to enable AWS access in your SparkSession. Pass as kwargs to `create_session`.

    :param credentials_provider: how to provide AWS credentials. Default gets them from the environment variables,
        but in production you might want to use IAM with 'com.amazonaws.auth.InstanceProfileCredentialsProvider'. See
        `here <https://hadoop.apache.org/docs/stable/hadoop-aws/tools/hadoop-aws/index.html#Authenticating_with_S3>`__
        for options.
    """
    # SparkConf args to add hadoop-aws dependency and allow advanced S3 access.
    # Note: this will not work if you use spark-submit, unless you explicitly specify the option
    # `spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.1 app.py`.
    if credentials_provider is None:
        credentials_provider = "com.amazonaws.auth.EnvironmentVariableCredentialsProvider"
    return {
        "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
        "spark.executor.extraJavaOptions": "-Dcom.amazonaws.services.s3.enableV4=true",
        "spark.driver.extraJavaOptions": "-Dcom.amazonaws.services.s3.enableV4=true",
        "com.amazonaws.auth.EnvironmentVariableCredentialsProvider": credentials_provider,
    }
