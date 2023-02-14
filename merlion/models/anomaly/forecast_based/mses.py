#
# Copyright (c) 2023 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
MSES (Multi-Scale Exponential Smoother) forecasting model adapted for anomaly detection.
"""
import pandas as pd

from merlion.models.anomaly.forecast_based.base import ForecastingDetectorBase
from merlion.models.anomaly.base import DetectorConfig
from merlion.models.forecast.smoother import MSESConfig, MSES, MSESTrainConfig
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils.time_series import TimeSeries


class MSESDetectorConfig(MSESConfig, DetectorConfig):
    """
    Configuration class for an MSES forecasting model adapted for anomaly detection.
    """

    _default_threshold = AggregateAlarms(alm_threshold=2)

    def __init__(self, max_forecast_steps: int, online_updates: bool = True, **kwargs):
        super().__init__(max_forecast_steps=max_forecast_steps, **kwargs)
        self.online_updates = online_updates


class MSESDetector(ForecastingDetectorBase, MSES):
    config_class = MSESDetectorConfig

    @property
    def online_updates(self):
        return self.config.online_updates

    @property
    def _default_train_config(self):
        return MSESTrainConfig(train_cadence=1 if self.online_updates else None)

    def get_anomaly_score(
        self, time_series: TimeSeries, time_series_prev: TimeSeries = None, exog_data=None
    ) -> TimeSeries:
        if self.online_updates:
            time_series, time_series_prev = self.transform_time_series(time_series, time_series_prev)
            if time_series_prev is None:
                full_ts = time_series
            else:
                full_ts = time_series_prev + time_series
            forecast, err = self.update(full_ts.to_pd(), train_cadence=pd.to_timedelta(0))
            forecast, err = [x.bisect(time_series.t0, t_in_left=False)[1] for x in [forecast, err]]
            return TimeSeries.from_pd(self.forecast_to_anom_score(time_series, forecast, err))
        else:
            return super().get_anomaly_score(time_series, time_series_prev)
