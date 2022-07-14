import unittest

import pandas as pd
import numpy as np

from merlion.models.forecast.prophet import Prophet, ProphetConfig
from merlion.utils.resample import to_timestamp


class TestProphet(unittest.TestCase):
    def test_resample_time_stamps(self):
        # arrange
        config = ProphetConfig()
        prophet = Prophet(config)
        prophet.last_train_time = pd._libs.tslibs.timestamps.Timestamp(year=2022, month=1, day=1)
        prophet.timedelta = pd._libs.tslibs.timedeltas.Timedelta(days=1)
        target = np.array([to_timestamp(pd._libs.tslibs.timestamps.Timestamp(year=2022, month=1, day=2))])

        # act
        output = prophet.resample_time_stamps(time_stamps=1)

        # assert
        assert output == target
