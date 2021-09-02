#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
"""
Contains the `ModelFactory`.
"""
import inspect
from typing import Type
import dill
from merlion.models.base import ModelBase
from merlion.utils import dynamic_import


import_alias = dict(
    # Default models
    DefaultDetector="merlion.models.defaults:DefaultDetector",
    DefaultForecaster="merlion.models.defaults:DefaultForecaster",
    # Anomaly detection models
    ArimaDetector="merlion.models.anomaly.forecast_based.arima:ArimaDetector",
    DynamicBaseline="merlion.models.anomaly.dbl:DynamicBaseline",
    IsolationForest="merlion.models.anomaly.isolation_forest:IsolationForest",
    ETSDetector="merlion.models.anomaly.forecast_based.ets:ETSDetector",
    LSTMDetector="merlion.models.anomaly.forecast_based.lstm:LSTMDetector",
    MSESDetector="merlion.models.anomaly.forecast_based.mses:MSESDetector",
    ProphetDetector="merlion.models.anomaly.forecast_based.prophet:ProphetDetector",
    RandomCutForest="merlion.models.anomaly.random_cut_forest:RandomCutForest",
    SarimaDetector="merlion.models.anomaly.forecast_based.sarima:SarimaDetector",
    WindStats="merlion.models.anomaly.windstats:WindStats",
    SpectralResidual="merlion.models.anomaly.spectral_residual:SpectralResidual",
    ZMS="merlion.models.anomaly.zms:ZMS",
    DeepPointAnomalyDetector="merlion.models.anomaly.deep_point_anomaly_detector:DeepPointAnomalyDetector",
    # Multivariate Anomaly Detection models
    AutoEncoder="merlion.models.anomaly.autoencoder:AutoEncoder",
    VAE="merlion.models.anomaly.vae:VAE",
    DAGMM="merlion.models.anomaly.dagmm:DAGMM",
    LSTMED="merlion.models.anomaly.lstm_ed:LSTMED",
    # Forecasting models
    Arima="merlion.models.forecast.arima:Arima",
    ETS="merlion.models.forecast.ets:ETS",
    LSTM="merlion.models.forecast.lstm:LSTM",
    MSES="merlion.models.forecast.smoother:MSES",
    Prophet="merlion.models.forecast.prophet:Prophet",
    Sarima="merlion.models.forecast.sarima:Sarima",
    StatThreshold="merlion.models.anomaly.stat_threshold:StatThreshold",
    VectorAR="merlion.models.forecast.vector_ar:VectorAR",
    RandomForestForecaster="merlion.models.forecast.baggingtrees:RandomForestForecaster",
    ExtraTreesForecaster="merlion.models.forecast.baggingtrees:ExtraTreesForecaster",
    LGBMForecaster="merlion.models.forecast.boostingtrees:LGBMForecaster",
    # Ensembles
    DetectorEnsemble="merlion.models.ensemble.anomaly:DetectorEnsemble",
    ForecasterEnsemble="merlion.models.ensemble.forecast:ForecasterEnsemble",
    MoE_ForecasterEnsemble="merlion.models.ensemble.MoE_forecast:MoE_ForecasterEnsemble",
    # Layers
    SeasonalityLayer="merlion.models.automl.seasonality_mixin:SeasonalityLayer",
    AutoSarima="merlion.models.automl.autosarima:AutoSarima",
)


class ModelFactory:
    @classmethod
    def get_model_class(cls, name: str) -> Type[ModelBase]:
        return dynamic_import(name, import_alias)

    @classmethod
    def create(cls, name, **kwargs) -> ModelBase:
        model_class = cls.get_model_class(name)
        signature = inspect.signature(model_class.__init__)
        if "config" in signature.parameters:
            config, kwargs = model_class.config_class.from_dict(kwargs, return_unused_kwargs=True)
            init_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}
            model = model_class(config, **init_kwargs)
            model._load_state({k: v for k, v in kwargs.items() if k not in init_kwargs})
        else:
            model = model_class(**kwargs)
        return model

    @classmethod
    def load(cls, name, model_path, **kwargs) -> ModelBase:
        if model_path is None:
            return cls.create(name, **kwargs)
        else:
            model_class = cls.get_model_class(name)
            return model_class.load(model_path, **kwargs)

    @classmethod
    def load_bytes(cls, obj, **kwargs) -> ModelBase:
        name = dill.loads(obj)[0]
        model_class = cls.get_model_class(name)
        return model_class.from_bytes(obj, **kwargs)
