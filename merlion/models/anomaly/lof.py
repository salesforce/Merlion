#
# Copyright (c) 2024 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
The classic LocalOutlierFactor model for anomaly detection.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from merlion.models.anomaly.base import DetectorConfig, DetectorBase
from merlion.transform.moving_average import DifferenceTransform
from merlion.transform.sequence import TransformSequence
from merlion.transform.resample import Shingle

logger = logging.getLogger(__name__)


class LOFConfig(DetectorConfig):
    """
    Configuration class for `LocalOutlierFactor`.
    """

    _default_transform = TransformSequence([DifferenceTransform(), Shingle(size=2, stride=1)])

    def __init__(
        self,
        n_neighbors=20,
        algorithm="auto",
        leaf_size=30,
        metric="minkowski",
        p=2,
        metric_params=None,
        contamination=0.1,
        n_jobs=1,
        novelty=True,
        **kwargs
    ):
        """
        n_neighbors : int, optional (default=20)
            Number of neighbors to use by default for `kneighbors` queries.
            If n_neighbors is larger than the number of samples provided,
            all samples will be used.

        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
            Algorithm used to compute the nearest neighbors:

            - 'ball_tree' will use BallTree
            - 'kd_tree' will use KDTree
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
              based on the values passed to :meth:`fit` method.

            Note: fitting on sparse input will override the setting of
            this parameter, using brute force.

        leaf_size : int, optional (default=30)
            Leaf size passed to `BallTree` or `KDTree`. This can
            affect the speed of the construction and query, as well as the memory
            required to store the tree. The optimal value depends on the
            nature of the problem.

        metric : string or callable, default 'minkowski'
            metric used for the distance computation. Any metric from scikit-learn
            or scipy.spatial.distance can be used.

            If 'precomputed', the training input X is expected to be a distance
            matrix.

            If metric is a callable function, it is called on each
            pair of instances (rows) and the resulting value recorded. The callable
            should take two arrays as input and return one value indicating the
            distance between them. This works for Scipy's metrics, but is less
            efficient than passing the metric name as a string.

            Valid values for metric are:

            - from scikit-learn: ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
              'manhattan']

            - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
              'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
              'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
              'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
              'sqeuclidean', 'yule']

            See the documentation for scipy.spatial.distance for details on these
            metrics:
            http://docs.scipy.org/doc/scipy/reference/spatial.distance.html

        p : integer, optional (default = 2)
            Parameter for the Minkowski metric from
            sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
            equivalent to using manhattan_distance (l1), and euclidean_distance
            (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
            See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances

        metric_params : dict, optional (default = None)
            Additional keyword arguments for the metric function.

        contamination : float in (0., 0.5), optional (default=0.1)
            The amount of contamination of the data set, i.e. the proportion
            of outliers in the data set. When fitting this is used to define the
            threshold on the decision function.

        n_jobs : int, optional (default = 1)
            The number of parallel jobs to run for neighbors search.
            If ``-1``, then the number of jobs is set to the number of CPU cores.
            Affects only kneighbors and kneighbors_graph methods.

        novelty : bool (default=False)
            By default, LocalOutlierFactor is only meant to be used for outlier
            detection (novelty=False). Set novelty to True if you want to use
            LocalOutlierFactor for novelty detection. In this case be aware that
            that you should only use predict, decision_function and score_samples
            on new unseen data and not on the training set.
        """
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.novelty = novelty
        # Expect the max_score be overridden in the calibrator function
        kwargs["max_score"] = 1.0
        super().__init__(**kwargs)


class LOF(DetectorBase):
    """
    The classic LocalOutlierFactor sklearn implementation.
    """

    config_class = LOFConfig

    def __init__(self, config: LOFConfig):
        super().__init__(config)
        self.model = LocalOutlierFactor(
            n_neighbors=config.n_neighbors,
            algorithm=config.algorithm,
            leaf_size=config.leaf_size,
            metric=config.metric,
            p=config.p,
            metric_params=config.metric_params,
            contamination=config.contamination,
            n_jobs=config.n_jobs,
            novelty=config.novelty,
        )

    @property
    def require_even_sampling(self) -> bool:
        return False

    @property
    def require_univariate(self) -> bool:
        return False

    def _train(self, train_data: pd.DataFrame, train_config=None) -> pd.DataFrame:
        times, train_values = train_data.index, train_data.values
        self.model.fit(train_values)
        train_scores = -self.model.score_samples(train_values)
        return pd.DataFrame(train_scores, index=times, columns=["anom_score"])

    def _get_anomaly_score(self, time_series: pd.DataFrame, time_series_prev: pd.DataFrame = None) -> pd.DataFrame:
        # Return the negative of model's score, since model scores are in [-1, 0), where more negative = more anomalous
        scores = -self.model.score_samples(np.array(time_series.values))
        return pd.DataFrame(scores, index=time_series.index)
