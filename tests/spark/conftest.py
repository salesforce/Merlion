#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import pytest
from pyspark import SparkConf
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    # Creates more helpful debug messages if Spark tests fail for some Java-related reason
    try:
        import faulthandler

        faulthandler.enable()
        faulthandler.disable()
    except:
        pass
    # Set timeout & heartbeat interval to 10 minutes to ensure tests can run to completion
    conf = SparkConf(False).setMaster("local[2]").setAppName("unit-tests")
    conf = conf.set("spark.network.timeout", "600000").set("spark.executor.heartbeatInterval", "600000")
    return SparkSession.builder.config(conf=conf).getOrCreate()
