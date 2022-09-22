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
    # Can result in more helpful debug messages if Spark tests fail for some Java-related reason
    try:
        import faulthandler

        faulthandler.enable()
        faulthandler.disable()
    except:
        pass
    return SparkSession.builder.master("local[1]").appName("unit-tests").getOrCreate()
