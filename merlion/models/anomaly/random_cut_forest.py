#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Wrapper around AWS's Random Cut Forest anomaly detection model.
"""
import bisect
import copy
import logging
from os.path import abspath, dirname, join, pathsep

import numpy as np
import pandas as pd
from py4j.java_gateway import JavaGateway

from merlion.models.anomaly.base import DetectorConfig, DetectorBase
from merlion.transform.moving_average import DifferenceTransform
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import Shingle
from merlion.post_process.threshold import AggregateAlarms
from merlion.utils import UnivariateTimeSeries, TimeSeries
from merlion.utils.resample import to_timestamp

logger = logging.getLogger(__name__)


class JVMSingleton:
    _gateway = None

    @classmethod
    def gateway(cls):
        resource_dir = join(dirname(dirname(dirname(abspath(__file__)))), "resources")
        jars = ["gson-2.8.9.jar", "randomcutforest-core-1.0.jar", "randomcutforest-serialization-json-1.0.jar"]
        classpath = pathsep.join(join(resource_dir, jar) for jar in jars)
        if cls._gateway is None:
            # --add-opens necessary to avoid exceptions in newer Java versions
            javaopts = ["--add-opens=java.base/java.util=ALL-UNNAMED", "--add-opens=java.base/java.nio=ALL-UNNAMED"]
            cls._gateway = JavaGateway.launch_gateway(classpath=classpath, javaopts=javaopts)
        return cls._gateway


class RandomCutForestConfig(DetectorConfig):
    """
    Configuration class for `RandomCutForest`. Refer to
    https://github.com/aws/random-cut-forest-by-aws/tree/main/Java for
    further documentation and defaults of the Java class.
    """

    _default_transform = TransformSequence([DifferenceTransform(), Shingle(size=5, stride=1)])

    def __init__(
        self,
        n_estimators: int = 100,
        parallel: bool = False,
        seed: int = None,
        max_n_samples: int = 512,
        thread_pool_size: int = 1,
        online_updates: bool = False,
        **kwargs
    ):
        """
        :param n_estimators: The number of trees in this forest.
        :param parallel: If true, then the forest will create an internal thread
            pool. Forest updates and traversals will be submitted to this thread
            pool, and individual trees will be updated or traversed in parallel.
            For larger shingle sizes, dimensions, and number of trees,
            parallelization may improve throughput.
            We recommend users benchmark against their target use case.
        :param seed: the random seed
        :param max_n_samples: The number of samples retained by by stream
            samplers in this forest.
        :param thread_pool_size: The number of threads to use in the internal
            thread pool.
        :param online_updates: Whether to update the model while running
            using it to evaluate new data.
        """
        self.n_estimators = n_estimators
        self.parallel = parallel
        self.seed = seed
        self.max_n_samples = max_n_samples
        self.thread_pool_size = thread_pool_size
        self.online_updates = online_updates
        kwargs["max_score"] = np.floor(np.log2(max_n_samples)) + 1
        super().__init__(**kwargs)

    @property
    def _default_threshold(self):
        if not self.enable_calibrator:
            return AggregateAlarms(alm_threshold=self.calibrator.max_score / 5)
        return AggregateAlarms(alm_threshold=3.0)

    @property
    def java_params(self):
        items = [
            ("numberOfTrees", self.n_estimators),
            ("randomSeed", self.seed),
            ("sampleSize", self.max_n_samples),
            ("threadPoolSize", self.thread_pool_size if self.parallel else None),
            ("parallelExecutionEnabled", self.parallel and self.thread_pool_size is not None),
        ]
        return {k: v for k, v in items if v is not None}


class RandomCutForest(DetectorBase):
    """
    The random cut forest is a refinement of the classic isolation forest
    algorithm. It was proposed in `Guha et al. 2016 <https://proceedings.mlr.press/v48/guha16.pdf>`__.
    """

    config_class = RandomCutForestConfig

    def __init__(self, config: RandomCutForestConfig):
        super().__init__(config)
        self.forest = None

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    @property
    def online_updates(self) -> bool:
        return self.config.online_updates

    def __getstate__(self):
        # Copy state, remove forest, and then deepcopy
        # (since we can't deepcopy the forest)
        state = copy.copy(self.__dict__)
        forest = state.pop("forest", None)
        state = copy.deepcopy(state)

        # Set the forest in the copied state to the serialized version
        # The transform is specified the config, so don't save it
        RCFSerDe = JVMSingleton.gateway().jvm.com.amazon.randomcutforest.serialize.RandomCutForestSerDe
        state["forest"] = str(RCFSerDe().toJson(forest))
        return state

    def __setstate__(self, state):
        # Remove the serialized forest from the state before setting it
        # Set the forest manually after deserializing it
        RCFSerDe = JVMSingleton.gateway().jvm.com.amazon.randomcutforest.serialize.RandomCutForestSerDe
        forest = RCFSerDe().fromJson(state.pop("forest", None))
        super().__setstate__(state)
        self.forest = forest

    def _forest_predict(self, data: np.ndarray, online_updates: bool):
        scores = []
        n, d = data.shape
        gateway = JVMSingleton.gateway()
        data_bytes = data.astype(dtype=">d").tobytes()
        data_jarray = gateway.new_array(gateway.jvm.double, n * d)
        gateway.jvm.java.nio.ByteBuffer.wrap(data_bytes).asDoubleBuffer().get(data_jarray)
        for i in range(n):
            jpoint = data_jarray[d * i : d * (i + 1)]
            scores.append(self.forest.getAnomalyScore(jpoint))
            if online_updates:
                self.forest.update(jpoint)
        return np.array(scores)

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        times, train_values = train_data.index, train_data.values

        # Initialize the RRCF, now that we know the dimension of the data
        JRCF = JVMSingleton.gateway().jvm.com.amazon.randomcutforest.RandomCutForest
        forest = JRCF.builder()
        forest = forest.dimensions(train_data.shape[1])
        for k, v in self.config.java_params.items():
            forest = getattr(forest, k)(v)
        self.forest = forest.build()

        train_scores = self._forest_predict(train_values, online_updates=True)
        return pd.DataFrame(train_scores, index=times, columns=["anom_score"])

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        if self.last_train_time is None:
            raise RuntimeError("train() must be called before you can invoke get_anomaly_score()")

        t0 = bisect.bisect_right(time_series.index, self.last_train_time)
        if 0 < t0 < len(time_series):
            old = self._forest_predict(time_series.values[:t0], False)
            new = self._forest_predict(time_series.values[t0:], self.online_updates)
            scores = np.concatenate((old, new))
        else:
            scores = self._forest_predict(time_series.values, self.online_updates and t0 > 0)
        if self.online_updates and t0 > 0:
            self.last_train_time = time_series.index[-1]

        return pd.DataFrame(scores, index=time_series.index)
