#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import logging
from os.path import abspath, dirname, join
import pytest
import sys
import unittest

import numpy as np
from merlion.utils import TimeSeries
from ts_datasets.forecast import SeattleTrail
from merlion.transform.normalize import MinMaxNormalize
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import TemporalResample
from merlion.transform.bound import LowerUpperClip
from merlion.transform.moving_average import DifferenceTransform
from merlion.models.forecast.seq_ar_common import gen_next_seq_label_pairs
from merlion.models.forecast.vector_ar import VectorAR, VectorARConfig

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestVectorAR(unittest.TestCase):
    """
    we test data loading, model instantiation, forecasting consistency, in particular
    (1) load a testing data
    (2) transform data
    (3) instantiate the VectorAR model and train
    (4) forecast, and the forecasting result agrees with the reference
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_forecast_steps = 3
        self.maxlags = 24 * 7
        self.i = 0
        # t = int(datetime(2019, 1, 1, 0, 0, 0).timestamp())

        dataset = "seattle_trail"
        d, md = SeattleTrail(rootdir=join(rootdir, "data", "multivariate", dataset))[0]
        t = int(d[md["trainval"]].index[-1].to_pydatetime().timestamp())
        data = TimeSeries.from_pd(d)
        cleanup_transform = TransformSequence(
            [TemporalResample(missing_value_policy="FFill"), LowerUpperClip(upper=300), DifferenceTransform()]
        )
        cleanup_transform.train(data)
        data = cleanup_transform(data)

        train_data, test_data = data.bisect(t)

        minmax_transform = MinMaxNormalize()
        minmax_transform.train(train_data)
        self.train_data_norm = minmax_transform(train_data)
        self.test_data_norm = minmax_transform(test_data)

        self.model = VectorAR(
            VectorARConfig(max_forecast_steps=self.max_forecast_steps, maxlags=self.maxlags, target_seq_index=self.i)
        )

    def run_test(self, univariate):
        logger.info("Training model...")
        if univariate:
            name = self.train_data_norm.names[self.i]
            self.model.config.maxlags = 7
            self.train_data_norm = self.train_data_norm.univariates[name][::24].to_ts()
            self.test_data_norm = self.test_data_norm.univariates[name][::24].to_ts()
            self.i = 0
        yhat, sigma = self.model.train(self.train_data_norm)
        logger.info("Model trained...")
        self.assertEqual(len(yhat), len(sigma))
        y = self.model.transform(self.train_data_norm).to_pd().iloc[:, self.i]
        # residual is y - yhat
        yhat = yhat.univariates[yhat.names[self.i]].np_values
        resid = self.model.model.resid
        resid = resid if univariate else resid.iloc[:, self.i]

        self.assertAlmostEqual(np.max(np.abs((y - yhat) - resid)), 0, places=6)
        self.assertEqual(len(self.model._forecast), self.max_forecast_steps)
        self.assertAlmostEqual(self.model._forecast.mean(), 0.5, delta=0.1)
        testing_data_gen = gen_next_seq_label_pairs(self.test_data_norm, self.i, self.maxlags, self.max_forecast_steps)
        testing_instance, testing_label = next(testing_data_gen)
        pred, err = self.model.forecast(testing_label.time_stamps, testing_instance)
        self.assertEqual(len(pred), self.max_forecast_steps)
        self.assertEqual(len(err), self.max_forecast_steps)
        pred = pred.univariates[pred.names[0]].np_values
        self.assertAlmostEqual(pred.mean(), 0.5, delta=0.1)

        pred2, _ = self.model.forecast(self.max_forecast_steps, testing_instance)
        pred2 = pred2.univariates[pred2.names[0]].np_values
        self.assertSequenceEqual(list(pred), list(pred2))

        logger.info("Testing save/load...")
        savedir = join(rootdir, "tmp", "autosarima")
        self.model.save(dirname=savedir)
        VectorAR.load(dirname=savedir)

    @pytest.mark.skip(reason="platform-specific segfaults")
    def test_forecast_univariate(self):
        self.run_test(True)

    def test_forecast_multivariate(self):
        self.run_test(False)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s", stream=sys.stdout, level=logging.DEBUG
    )
    unittest.main()
