#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark_session():
    spark = SparkSession.builder.master("local[2]").appName("unit-tests").getOrCreate()
    return spark
