import logging
from os.path import abspath, dirname, join
import pytest
import sys
import unittest

import numpy as np
import pandas as pd

from merlion.evaluate.forecast import ForecastMetric
from merlion.models.automl.autosarima import AutoSarima, AutoSarimaConfig
from merlion.models.automl.seasonality_mixin import SeasonalityLayer
from merlion.models.forecast.sarima import Sarima
from merlion.utils import TimeSeries, autosarima_utils

logger = logging.getLogger(__name__)
rootdir = dirname(dirname(dirname(abspath(__file__))))


class TestAutoSarima(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_data = np.array(
            [
                605.0,
                586.0,
                586.0,
                559.0,
                511.0,
                443.0,
                422.0,
                395.0,
                382.0,
                370.0,
                383.0,
                397.0,
                420.0,
                455.0,
                493.0,
                554.0,
                610.0,
                666.0,
                715.0,
                755.0,
                778.0,
                794.0,
                806.0,
                808.0,
                776.0,
                723.0,
                709.0,
                660.0,
                585.0,
                527.0,
                462.0,
                437.0,
                413.0,
                407.0,
                404.0,
                420.0,
                441.0,
                471.0,
                526.0,
                571.0,
                612.0,
                635.0,
                613.0,
                608.0,
                614.0,
                637.0,
                669.0,
                683.0,
                687.0,
                660.0,
                661.0,
                632.0,
                573.0,
                521.0,
                481.0,
                452.0,
                447.0,
                425.0,
                427.0,
                441.0,
                438.0,
                472.0,
                528.0,
                596.0,
                661.0,
                708.0,
                754.0,
                781.0,
                808.0,
                819.0,
                820.0,
                801.0,
                770.0,
                717.0,
                697.0,
                655.0,
                607.0,
                552.0,
                512.0,
                475.0,
                452.0,
                436.0,
                429.0,
                433.0,
                430.0,
                472.0,
                536.0,
                611.0,
                662.0,
                705.0,
                707.0,
                718.0,
                733.0,
                741.0,
                737.0,
                710.0,
                647.0,
                593.0,
                564.0,
                528.0,
                507.0,
                466.0,
                428.0,
                396.0,
                382.0,
                371.0,
                374.0,
                372.0,
                373.0,
                406.0,
                451.0,
                480.0,
                487.0,
                487.0,
                494.0,
                505.0,
                519.0,
                535.0,
                560.0,
                570.0,
                551.0,
                541.0,
                552.0,
                521.0,
                477.0,
                426.0,
                395.0,
                370.0,
                357.0,
                349.0,
                362.0,
                374.0,
                402.0,
                427.0,
                456.0,
                500.0,
                558.0,
                608.0,
                661.0,
                704.0,
                736.0,
                763.0,
                782.0,
                782.0,
                775.0,
                730.0,
                716.0,
                657.0,
                590.0,
                518.0,
                469.0,
                438.0,
                418.0,
                406.0,
                405.0,
                418.0,
                440.0,
                467.0,
                526.0,
                588.0,
                660.0,
                713.0,
                767.0,
                800.0,
                828.0,
                837.0,
                851.0,
                836.0,
                813.0,
                776.0,
                753.0,
                695.0,
                620.0,
                542.0,
                494.0,
                459.0,
                439.0,
                417.0,
                423.0,
                429.0,
                450.0,
                482.0,
                539.0,
                613.0,
                677.0,
                736.0,
                777.0,
                820.0,
                828.0,
                838.0,
                824.0,
                805.0,
                774.0,
                736.0,
                718.0,
                666.0,
                601.0,
                532.0,
                485.0,
                449.0,
                427.0,
                415.0,
                418.0,
                438.0,
                445.0,
                487.0,
                535.0,
                598.0,
                669.0,
                730.0,
                776.0,
                801.0,
                824.0,
                826.0,
                819.0,
                802.0,
                771.0,
                740.0,
                722.0,
                676.0,
                603.0,
                546.0,
                488.0,
                458.0,
                438.0,
                421.0,
                419.0,
                426.0,
                450.0,
                487.0,
                550.0,
                612.0,
                683.0,
                739.0,
                785.0,
                818.0,
                837.0,
                847.0,
                848.0,
                834.0,
                810.0,
                764.0,
                746.0,
                698.0,
                634.0,
                584.0,
                519.0,
                490.0,
                463.0,
                446.0,
                439.0,
                436.0,
                442.0,
                483.0,
                566.0,
                643.0,
                723.0,
                773.0,
                822.0,
                849.0,
                865.0,
                881.0,
                882.0,
                872.0,
                840.0,
                797.0,
                773.0,
                724.0,
                661.0,
                593.0,
                547.0,
                512.0,
                484.0,
                463.0,
                452.0,
                442.0,
                447.0,
                506.0,
                598.0,
                692.0,
                756.0,
                804.0,
                836.0,
                866.0,
                886.0,
                890.0,
                878.0,
                863.0,
                830.0,
                770.0,
                756.0,
                704.0,
                629.0,
                566.0,
                518.0,
                484.0,
                461.0,
                450.0,
                446.0,
                467.0,
                485.0,
                525.0,
                585.0,
                670.0,
                747.0,
                805.0,
                839.0,
                857.0,
                844.0,
                834.0,
                828.0,
                834.0,
                815.0,
                781.0,
                762.0,
                710.0,
                641.0,
                571.0,
                527.0,
                498.0,
                475.0,
                466.0,
                472.0,
                478.0,
                506.0,
                542.0,
                594.0,
                661.0,
                728.0,
                786.0,
                828.0,
                846.0,
                854.0,
                852.0,
                862.0,
                859.0,
                853.0,
                815.0,
                802.0,
                749.0,
                679.0,
                604.0,
                569.0,
                529.0,
                513.0,
                500.0,
                495.0,
                511.0,
                525.0,
                557.0,
                605.0,
                676.0,
                737.0,
                791.0,
                837.0,
                856.0,
                886.0,
                901.0,
                916.0,
                905.0,
                871.0,
                823.0,
                798.0,
                738.0,
                666.0,
                593.0,
                553.0,
                515.0,
                494.0,
                487.0,
                482.0,
                498.0,
                515.0,
                540.0,
                584.0,
                633.0,
                705.0,
                754.0,
                801.0,
                833.0,
                863.0,
                872.0,
                867.0,
                845.0,
                819.0,
                787.0,
                762.0,
                717.0,
                651.0,
                575.0,
                537.0,
                500.0,
                470.0,
                457.0,
                465.0,
                467.0,
                488.0,
                525.0,
                581.0,
                654.0,
                732.0,
                789.0,
                826.0,
                852.0,
                870.0,
                859.0,
                853.0,
                820.0,
                789.0,
                741.0,
                731.0,
                683.0,
                620.0,
                566.0,
                514.0,
                478.0,
                462.0,
                435.0,
                432.0,
                434.0,
                441.0,
                483.0,
                563.0,
                644.0,
                729.0,
                798.0,
                847.0,
                874.0,
                890.0,
                908.0,
                902.0,
                895.0,
                860.0,
                815.0,
                792.0,
                732.0,
                672.0,
                605.0,
                554.0,
                517.0,
                492.0,
                473.0,
                452.0,
                458.0,
                452.0,
                506.0,
                607.0,
                691.0,
                761.0,
                820.0,
                860.0,
                867.0,
                852.0,
                820.0,
                768.0,
                729.0,
                689.0,
                661.0,
                652.0,
                612.0,
                559.0,
                503.0,
                468.0,
                441.0,
                423.0,
                412.0,
                424.0,
                441.0,
                460.0,
                494.0,
                548.0,
                609.0,
                694.0,
                758.0,
                817.0,
                863.0,
                881.0,
                900.0,
                909.0,
                900.0,
                875.0,
                830.0,
                812.0,
                758.0,
                688.0,
                607.0,
                570.0,
                536.0,
                504.0,
                487.0,
                495.0,
                503.0,
                521.0,
                554.0,
                612.0,
                680.0,
                753.0,
                817.0,
                863.0,
                889.0,
                883.0,
                867.0,
                840.0,
                810.0,
                779.0,
                744.0,
                722.0,
                668.0,
                613.0,
                550.0,
                509.0,
                481.0,
                460.0,
                458.0,
                456.0,
                474.0,
                488.0,
                527.0,
                583.0,
                652.0,
                723.0,
                796.0,
                846.0,
                877.0,
                881.0,
                884.0,
                857.0,
                835.0,
                809.0,
                766.0,
                755.0,
                703.0,
                635.0,
                577.0,
                533.0,
                504.0,
                485.0,
                473.0,
                474.0,
                494.0,
                507.0,
                545.0,
                599.0,
                672.0,
                740.0,
                790.0,
                823.0,
                822.0,
                817.0,
                801.0,
                792.0,
                775.0,
                735.0,
                723.0,
                699.0,
                658.0,
                598.0,
                547.0,
                503.0,
                474.0,
                459.0,
                450.0,
                450.0,
                469.0,
                488.0,
                520.0,
                566.0,
                640.0,
                705.0,
                762.0,
                808.0,
                838.0,
                820.0,
                784.0,
                753.0,
                739.0,
                720.0,
                690.0,
                678.0,
                634.0,
                587.0,
                537.0,
                492.0,
                464.0,
                443.0,
                427.0,
                424.0,
                430.0,
                424.0,
                473.0,
                537.0,
                616.0,
                684.0,
                761.0,
                793.0,
                826.0,
                833.0,
                835.0,
                838.0,
                823.0,
                795.0,
                750.0,
                739.0,
                679.0,
                622.0,
                558.0,
                513.0,
                476.0,
                449.0,
                437.0,
                422.0,
                423.0,
                415.0,
                475.0,
                553.0,
                624.0,
                680.0,
                720.0,
                769.0,
                805.0,
                828.0,
                836.0,
                849.0,
                844.0,
                808.0,
                757.0,
                730.0,
                670.0,
                594.0,
                528.0,
                474.0,
                447.0,
                423.0,
                412.0,
                413.0,
                431.0,
                449.0,
                489.0,
                544.0,
                610.0,
                696.0,
                765.0,
                813.0,
                851.0,
                872.0,
                883.0,
                899.0,
                897.0,
                871.0,
                831.0,
                813.0,
                749.0,
                664.0,
                550.0,
                544.0,
                505.0,
                483.0,
                469.0,
                466.0,
                487.0,
                492.0,
                531.0,
                583.0,
                659.0,
                743.0,
                811.0,
                863.0,
                898.0,
                914.0,
                920.0,
                926.0,
                919.0,
                887.0,
                862.0,
                829.0,
                769.0,
                691.0,
                618.0,
                563.0,
                529.0,
                504.0,
                489.0,
                487.0,
                508.0,
                513.0,
                555.0,
                606.0,
                676.0,
                761.0,
                837.0,
                878.0,
                890.0,
                879.0,
                847.0,
                820.0,
                790.0,
                784.0,
                752.0,
                739.0,
                684.0,
            ]
        )
        test_data = np.array(
            [
                619.0,
                565.0,
                532.0,
                495.0,
                481.0,
                467.0,
                473.0,
                488.0,
                501.0,
                534.0,
                576.0,
                639.0,
                712.0,
                772.0,
                830.0,
                880.0,
                893.0,
                896.0,
                891.0,
                854.0,
                803.0,
                769.0,
                751.0,
                701.0,
                635.0,
                572.0,
                532.0,
                493.0,
                477.0,
                468.0,
                464.0,
                477.0,
                492.0,
                519.0,
                568.0,
                624.0,
                696.0,
                761.0,
                812.0,
                836.0,
                838.0,
                829.0,
                807.0,
                785.0,
                756.0,
                719.0,
                703.0,
                659.0,
            ]
        )
        data = np.concatenate([train_data, test_data])
        data = TimeSeries.from_pd(pd.Series(data))
        self.train_data = data[: len(train_data)]
        self.test_data = data[len(train_data) :]
        self.max_forecast_steps = len(self.test_data)
        self.model = SeasonalityLayer(
            model=AutoSarima(
                model=Sarima(
                    AutoSarimaConfig(
                        max_forecast_steps=self.max_forecast_steps, maxiter=5
                    )
                )
            )
        )

    def test_forecast(self):
        # sMAPE = 3.4491
        train_pred, train_err = self.model.train(
            self.train_data,
            train_config={
                "enforce_stationarity": False,
                "enforce_invertibility": False,
            },
        )

        # check automatic periodicity detection
        k = self.test_data.names[0]
        m = autosarima_utils.multiperiodicity_detection(
            self.train_data.univariates[k].np_values
        )
        self.assertEqual(m[0], 24)

        # check the length of training results
        self.assertEqual(len(train_pred), len(train_err))

        # check the length of forecasting results
        pred, err = self.model.forecast(self.max_forecast_steps)
        self.assertEqual(len(pred), self.max_forecast_steps)
        self.assertEqual(len(err), self.max_forecast_steps)

        # check the forecasting results w.r.t sMAPE
        y_true = self.test_data.univariates[k].np_values
        y_hat = pred.univariates[pred.names[0]].np_values
        smape = np.mean(
            200.0 * np.abs((y_true - y_hat) / (np.abs(y_true) + np.abs(y_hat)))
        )
        logger.info(f"sMAPE = {smape:.4f}")
        self.assertLessEqual(smape, 4.5)

        # check smape in evalution
        smape_compare = ForecastMetric.sMAPE.value(self.test_data, pred)
        self.assertAlmostEqual(smape, smape_compare)

        # test save/load
        savedir = join(rootdir, "tmp", "autosarima")
        self.model.save(dirname=savedir)
        SeasonalityLayer.load(dirname=savedir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        stream=sys.stdout,
        level=logging.DEBUG,
    )
    unittest.main()
