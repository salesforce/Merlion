#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from os.path import abspath, dirname, join
import sys
import unittest

import pandas as pd
import numpy as np

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.automl.autoets import AutoETS, AutoETSConfig
from merlion.models.forecast.ets import ETS, ETSConfig
from merlion.utils.time_series import TimeSeries, to_pd_datetime

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestETS(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_data = np.array(
            [
                49749475.08,
                48334704.82,
                48275157.57,
                43969281.56,
                46870666.51,
                45924937.95,
                44988678.08,
                44133701.02,
                50423887.67,
                47365181.62,
                45183562.86,
                44733997.96,
                43705038.61,
                48503231.02,
                45329947.95,
                45119948.58,
                47757383.6,
                50188444.77,
                47826347.76,
                47621799.23,
                46608866.38,
                48917390.8,
                47899499.45,
                46243813.08,
                44888625.35,
                44630086.18,
                48204925.76,
                46464361.88,
                47060765.68,
                45909681.68,
                47194223.17,
                45633673.84,
                43081657.51,
                41358352.79,
                42239826.37,
                45102996.78,
                43149465.36,
                43066645.59,
                43602653.58,
                45782124.16,
                46124776.79,
                45125584.18,
                65821010.24,
                49909055.87,
                55666423.99,
                61820270.05,
                80931028.42,
                40432678.46,
                42775937.09,
                40673653.3,
                40654715.99,
                39599811.1,
                46153156.48,
                47336192.79,
                48716033.27,
                44125860.84,
                46980667.15,
                44627510.46,
                44872297.41,
                42876467.62,
                43459171.87,
                45887401.96,
                44973284.11,
                48676717.62,
                43529990.87,
                46861899.62,
                45446059.32,
                44046586.63,
                45293460.06,
                48772039.02,
                47669719.65,
                47447526.22,
                45884025.5,
                47578624.19,
                47859258.4,
                45515914.52,
                45274339.01,
                43683243.92,
                48015073.36,
                46249598.39,
                46917127.1,
                47416905.45,
                45376432.48,
                46763000.79,
                43792604.96,
                42716522.23,
                42195007.11,
                47211092.36,
                44374292.45,
                45819133.11,
                45855520.28,
                48654605.42,
                48474212.41,
                46438802.12,
                66565730.02,
                49384577.42,
                55558067.21,
                60082608.34,
                76994571.24,
                46041807.5,
                44955394.34,
                42022604.47,
                42080268.78,
                39834180.38,
                46085308.65,
                50007132.23,
                50195096.99,
                45770682.88,
                46860478.29,
                47479695.77,
                46900921.15,
                44993618.49,
                45272511.84,
                53502282.02,
                46629165.55,
                45072409.05,
                43716752.65,
                47124072.67,
                46925658.31,
                46823296.04,
                47892700.9,
                48281635.03,
                49651162.2,
                48412125.56,
                47668135.12,
                46597053.1,
                51253099.81,
                46099747.96,
                46059299.15,
                44097050.8,
            ]
        )
        test_data = np.array(
            [
                47485712.35,
                47403448.22,
                47355041.71,
                47447284.43,
                47159585.95,
                48329845.27,
                44225749.26,
                44354360.72,
                43734839.24,
                47566467.14,
                46128371.47,
                45122294.9,
                45544036.17,
            ]
        )
        self.train_data = TimeSeries.from_pd(pd.Series(train_data))
        self.test_data = TimeSeries.from_pd(
            pd.Series(
                test_data,
                index=pd.RangeIndex(start=len(self.train_data), stop=len(self.train_data) + test_data.shape[0]),
            )
        )
        self.max_forecast_steps = len(self.test_data)
        self.autoets_model = AutoETS(AutoETSConfig(pval=0.1, max_lag=55, max_forecast_steps=self.max_forecast_steps))
        self.ets_model = ETS(ETSConfig(seasonal_periods=4))

    def test_forecast(self):
        _, _ = self.autoets_model.train(self.train_data)
        forecast, lb, ub = self.autoets_model.forecast(self.max_forecast_steps, return_iqr=True)
        smape_auto = ForecastMetric.sMAPE.value(self.test_data, forecast, target_seq_index=0)
        logger.info(f"sMAPE = {smape_auto:.4f} for {self.max_forecast_steps} step forecasting for AutoETS")
        self.assertAlmostEqual(smape_auto, 2.21, delta=1)
        self.autoets_model.save(join(rootdir, "tmp", "autoets"))

        _, _ = self.ets_model.train(self.train_data)
        forecast, lb, ub = self.ets_model.forecast(self.max_forecast_steps, return_iqr=True)
        smape = ForecastMetric.sMAPE.value(self.test_data, forecast, target_seq_index=0)
        logger.info(f"sMAPE = {smape:.4f} for {self.max_forecast_steps} step forecasting for ETS")
        self.assertAlmostEqual(smape, 3.96, delta=1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
