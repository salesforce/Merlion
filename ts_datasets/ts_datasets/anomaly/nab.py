#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import datetime
import glob
import json
import logging
import os
import requests

import numpy as np
import pandas as pd
import tqdm

from ts_datasets.anomaly.base import TSADBaseDataset

logger = logging.getLogger(__name__)


class NAB(TSADBaseDataset):
    """
    Wrapper to load datasets found in the Numenta Anomaly Benchmark
    (https://github.com/numenta/NAB).

    The NAB contains a range of datasets and are categorized by their domains.
    """

    valid_subsets = [
        "all",
        "artificial",
        "artificialWithAnomaly",
        "realAWSCloudwatch",
        "realAdExchange",
        "realKnownCause",
        "realTraffic",
        "realTweets",
    ]

    def __init__(self, subset="all", rootdir=None):
        """
        :param subset: One of the elements in subsets.
        :param rootdir: The root directory at which the dataset can be found.
        """
        super().__init__()
        assert subset in self.valid_subsets, f"subset should be in {self.valid_subsets}, but got {subset}"
        self.subset = subset

        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "nab")

        if subset == "artificial":
            subsets = ["artificialNoAnomaly", "artificialWithAnomaly"]
        elif subset == "all":
            subsets = [
                "artificialNoAnomaly",
                "artificialWithAnomaly",
                "realAWSCloudwatch",
                "realAdExchange",
                "realKnownCause",
                "realTraffic",
                "realTweets",
            ]
        else:
            subsets = [subset]
        self.download(rootdir, subsets)
        dsetdirs = [os.path.join(rootdir, s) for s in subsets]

        labelfile = os.path.join(rootdir, "labels/combined_windows.json")
        with open(labelfile) as json_file:
            label_list = json.load(json_file)

        csvs = sum([sorted(glob.glob(f"{d}/*.csv")) for d in dsetdirs], [])
        for i, csv in enumerate(sorted(csvs)):
            df = pd.read_csv(csv)
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.sort_values(by="timestamp")
            if len(df["timestamp"][df["timestamp"].diff() == datetime.timedelta(0)]) != 0:
                df = df.drop_duplicates(subset="timestamp", keep="first")
                logger.warning(f"Time series {csv} (index {i}) has timestamp duplicates. Kept first values.")

            all_dt = np.unique(np.diff(df["timestamp"])).astype(int)
            gcd_dt = all_dt[0]
            for dt in all_dt[1:]:
                gcd_dt = np.gcd(gcd_dt, dt)
            gcd_dt = pd.to_timedelta(gcd_dt)
            labels = self.load_labels(csv, label_list, gcd_dt)
            df["anomaly"] = df["timestamp"].apply(lambda x: x in labels)
            df = df.set_index("timestamp")

            # First 15% of data is "probationary", i.e. model can use it to
            # warm-start without being tested. See Figure 2 of the NAB
            # paper https://arxiv.org/pdf/1510.03336.pdf
            n = len(df) * 0.15
            df["trainval"] = pd.Series(np.arange(len(df)) <= n, index=df.index)

            md_cols = ["anomaly", "trainval"]
            self.metadata.append(df[md_cols])
            self.time_series.append(df[[c for c in df.columns if c not in md_cols]])

    @staticmethod
    def load_labels(datafile, label_list, freq):
        filename = "/".join(datafile.split("/")[-2:])
        label_list = label_list[filename]
        labels = pd.DatetimeIndex([])
        for lp in label_list:
            start = pd.to_datetime(lp[0])
            end = pd.to_datetime(lp[1])
            labels = labels.append(pd.date_range(start=start, end=end, freq=freq))
        return labels

    @property
    def max_lead_sec(self):
        """
        The anomalies in the NAB dataset are already windows which permit early
        detection. So we explicitly disallow any earlier detection.
        """
        return 0

    def download(self, rootdir, subsets):
        csvs = [
            "artificialNoAnomaly/art_daily_no_noise.csv",
            "artificialNoAnomaly/art_daily_perfect_square_wave.csv",
            "artificialNoAnomaly/art_daily_small_noise.csv",
            "artificialNoAnomaly/art_flatline.csv",
            "artificialNoAnomaly/art_noisy.csv",
            "artificialWithAnomaly/art_daily_flatmiddle.csv",
            "artificialWithAnomaly/art_daily_jumpsdown.csv",
            "artificialWithAnomaly/art_daily_jumpsup.csv",
            "artificialWithAnomaly/art_daily_nojump.csv",
            "artificialWithAnomaly/art_increase_spike_density.csv",
            "artificialWithAnomaly/art_load_balancer_spikes.csv",
            "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
            "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
            "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
            "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv",
            "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
            "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv",
            "realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv",
            "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv",
            "realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv",
            "realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv",
            "realAWSCloudwatch/ec2_network_in_257a54.csv",
            "realAWSCloudwatch/ec2_network_in_5abac7.csv",
            "realAWSCloudwatch/elb_request_count_8c0756.csv",
            "realAWSCloudwatch/grok_asg_anomaly.csv",
            "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
            "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
            "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv",
            "realAdExchange/exchange-2_cpc_results.csv",
            "realAdExchange/exchange-2_cpm_results.csv",
            "realAdExchange/exchange-3_cpc_results.csv",
            "realAdExchange/exchange-3_cpm_results.csv",
            "realAdExchange/exchange-4_cpc_results.csv",
            "realAdExchange/exchange-4_cpm_results.csv",
            "realKnownCause/ambient_temperature_system_failure.csv",
            "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
            "realKnownCause/ec2_request_latency_system_failure.csv",
            "realKnownCause/machine_temperature_system_failure.csv",
            "realKnownCause/nyc_taxi.csv",
            "realKnownCause/rogue_agent_key_hold.csv",
            "realKnownCause/rogue_agent_key_updown.csv",
            "realTraffic/TravelTime_387.csv",
            "realTraffic/TravelTime_451.csv",
            "realTraffic/occupancy_6005.csv",
            "realTraffic/occupancy_t4013.csv",
            "realTraffic/speed_6005.csv",
            "realTraffic/speed_7578.csv",
            "realTraffic/speed_t4013.csv",
            "realTweets/Twitter_volume_AAPL.csv",
            "realTweets/Twitter_volume_AMZN.csv",
            "realTweets/Twitter_volume_CRM.csv",
            "realTweets/Twitter_volume_CVS.csv",
            "realTweets/Twitter_volume_FB.csv",
            "realTweets/Twitter_volume_GOOG.csv",
            "realTweets/Twitter_volume_IBM.csv",
            "realTweets/Twitter_volume_KO.csv",
            "realTweets/Twitter_volume_PFE.csv",
            "realTweets/Twitter_volume_UPS.csv",
        ]

        labelfile = "labels/combined_windows.json"
        path = os.path.join(rootdir, labelfile)
        if not os.path.isfile(path):
            print("Downloading label file...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            url = f"https://github.com/numenta/NAB/raw/master/{labelfile}"
            r = requests.get(url, stream=True)
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        f.flush()

        csvs = [f for f in csvs if not os.path.isfile(os.path.join(rootdir, f)) and f.split("/")[0] in subsets]
        for csv in tqdm.tqdm(csvs, desc="NAB Download", disable=len(csvs) == 0):
            path = os.path.join(rootdir, csv)
            if not os.path.isfile(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                url = f"https://github.com/numenta/NAB/raw/master/data/{csv}"
                r = requests.get(url, stream=True)
                with open(path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            f.flush()
