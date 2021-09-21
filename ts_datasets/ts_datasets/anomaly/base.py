#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import numpy as np
import pandas as pd

from ts_datasets.base import BaseDataset, _main_fns_docstr

_intro_docstr = """
Base dataset class for storing time series intended for anomaly detection.
"""

_extra_note = """

.. note::

    For each time series, the ``metadata`` will always have the key ``anomaly``, which is a 
    ``pd.Series`` of ``bool`` indicating whether each timestamp is anomalous.
"""


class TSADBaseDataset(BaseDataset):
    __doc__ = _intro_docstr + _main_fns_docstr + _extra_note

    @property
    def max_lead_sec(self):
        """
        The maximum number of seconds an anomaly may be detected early, for
        this dataset. ``None`` signifies no early detections allowed, or that
        the user may override this value with something better suited for their
        purposes.
        """
        return None

    @property
    def max_lag_sec(self):
        """
        The maximum number of seconds after the start of an anomaly, that we
        consider detections to be accurate (and not ignored for being too late).
        ``None`` signifies that any detection in the window is acceptable, or
        that the user may override this value with something better suited for
        their purposes.
        """
        return None

    def describe(self):
        anom_bds = []
        anom_locs = []
        anom_in_trainval = []
        for ts, md in self:
            boundaries = md.anomaly.iloc[1:] != md.anomaly.values[:-1]
            boundaries = boundaries[boundaries].index
            if len(boundaries) == 0:
                continue

            ts_len = ts.index[-1] - ts.index[0]
            if md.anomaly.iloc[0]:
                anom_bds.append((ts.index[0], boundaries[0]))
                anom_locs.append((boundaries[0] - ts.index[0]) / ts_len)
                anom_in_trainval.append(True)

            for t0, tf in zip(boundaries[:-1], boundaries[1:]):
                if md.anomaly[t0]:
                    anom_bds.append((t0, tf))
                    anom_locs.append((tf - ts.index[0]) / ts_len)
                    anom_in_trainval.append(bool(md.trainval[t0]))

            if md.anomaly[boundaries[-1]]:
                anom_bds.append((boundaries[-1], ts.index[-1]))
                anom_locs.append(1.0)
                anom_in_trainval.append(False)

        print("=" * 80)
        print(f"Time series in dataset have average length {int(np.mean([len(ts) for ts, md in self]))}.")
        print(f"Time series in dataset have {len(anom_bds) / len(self):.1f} anomalies on average.")
        print(
            f"{sum(anom_in_trainval) / len(anom_in_trainval) * 100:.1f}% of "
            f"anomalies are in the train/val split of their respective time "
            f"series."
        )
        print(f"Anomalies in dataset have average length {pd.Timedelta(np.mean([(tf - t0) for t0, tf in anom_bds]))}.")
        print(
            f"Average anomaly occurs {np.mean(anom_locs) * 100:.1f}% "
            f"(+/- {np.std(anom_locs) * 100:.1f}%) of the way through "
            f"its respective time series."
        )
        print("=" * 80)
