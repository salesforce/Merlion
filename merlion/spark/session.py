#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from pyspark import SparkConf
from pyspark.sql import SparkSession


def create_session():
    # Create spark config with PyArrow enabled
    conf = SparkConf()
    conf.setAppName("merlion")
    conf.set("spark.dynamicAllocation.enabled", "true")
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    # Add hadoop-aws dependency and allow advanced S3 access.
    # Note: this will not work if you use spark-submit, unless you explicitly specify the option
    # `spark-submit --packages org.apache.hadoop:hadoop-aws:3.3.1 app.py`.
    conf.set("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
    conf.set("spark.executor.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true")
    conf.set("spark.driver.extraJavaOptions", "-Dcom.amazonaws.services.s3.enableV4=true")

    # Set AWS credentials from environment variables.
    # FIXME: In production, you would probably want to use IAM w/ InstanceProfileCredentialsProvider.
    conf.set(
        "spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.EnvironmentVariableCredentialsProvider"
    )

    # Create spark session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark
