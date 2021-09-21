#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
from os.path import abspath, dirname, join
from collections import OrderedDict
import os

import numpy as np
from math import floor, ceil

from merlion.utils.time_series import ts_to_csv
from merlion.utils.ts_generator import GeneratorConcatenator, TimeSeriesGenerator
from merlion.transform.anomalize import LevelShift, Shock, TrendChange

MERLION_ROOT = dirname(dirname(dirname(abspath(__file__))))
DATADIR = join(MERLION_ROOT, "data")


def main():
    np.random.seed(12345)
    n = 10000

    # Generate Synthetic Time Series
    ts_generators = [
        # generates a time series that trends upward before
        # trending downward
        GeneratorConcatenator(
            generators=[
                # upward trend
                TimeSeriesGenerator(f=lambda x: x ** 1.6, n=floor(0.6 * n)),
                # downward trend
                TimeSeriesGenerator(f=lambda x: -x ** 1.2, n=ceil(0.4 * n)),
            ],
            noise=lambda: np.random.normal(0, 500),
            string_outputs=True,
            name="upward_downward",
        ),
        # generates a white noise series
        TimeSeriesGenerator(f=lambda x: 0, n=n, name="horizontal"),
        # generates a time series with multiple seasonality
        TimeSeriesGenerator(f=lambda x: 2 * np.sin(x * 0.1) + np.sin(x * 0.02), n=n, name="seasonal"),
    ]

    ts_list = [generator.generate(return_ts=True) for generator in ts_generators]

    # Initialize Anomaly Injection Transforms
    anomalize_kwargs = dict(anom_prob=0.002, anom_width_range=(20, 200), alpha=0.5)

    anomalizers = OrderedDict(
        shock=Shock(pos_prob=0.5, sd_range=(4, 8), **anomalize_kwargs),
        spike=Shock(pos_prob=1.0, sd_range=(4, 8), **anomalize_kwargs),
        dip=Shock(pos_prob=0.0, sd_range=(4, 8), **anomalize_kwargs),
        level=LevelShift(pos_prob=0.5, sd_range=(3, 6), **anomalize_kwargs),
        trend=TrendChange(anom_prob=0.01, pos_prob=0.5, scale_range=(2.5, 5)),
    )

    # make directory for writing anomalized data
    anom_dir = join(DATADIR, "synthetic_anomaly")
    os.makedirs(anom_dir, exist_ok=True)

    for i, ts in enumerate(ts_list):
        # write original ts
        csv = join(anom_dir, f"{ts.names[0]}.csv")
        ts_to_csv(ts, csv)
        # anomalize ts with each anomalizer
        for j, (name, anom) in enumerate(anomalizers.items()):
            np.random.seed(1000 * i + j)
            anom_ts = anom(ts)
            csv = join(anom_dir, f"{anom_ts.names[0]}_{name}_anomaly.csv")
            ts_to_csv(anom_ts, csv)


if __name__ == "__main__":
    main()
