This file outlines how you should navigate the Jupyter notebooks in this folder.
All new users should start with [`TimeSeries.ipynb`](TimeSeries.ipynb), which explains
how to use Merlion's `UnivariateTimeSeries` and `TimeSeries` classes. These classes are
the core data format used throughout the repo. 

If you are interested in anomaly detection, you should next read 
[`anomaly/AnomalyIntro.ipynb`](anomaly/0_AnomalyIntro.ipynb) to understand how to use
anomaly detection models in Merlion. Afterwards, if you want to implement a new
anomaly detection model in Merlion, please read [`CONTRIBUTING.md`](../CONTRIBUTING.md)
and [`anomaly/AnomalyNewModel.ipynb`](anomaly/3_AnomalyNewModel.ipynb).

If you are interested in forecasting, you should next read
[`forecast/ForecastIntro.ipynb`](forecast/0_ForecastIntro.ipynb) to understand how to use
forecasting models in Merlion. Afterward, if you want to implement a new forecasting
model in Merlion, please read [`CONTRIBUTING.md`](../CONTRIBUTING.md) and
and [`forecast/ForecastNewModel.ipynb`](forecast/3_ForecastNewModel.ipynb).

We offer more advanced tutorials on specific high-performing models (AutoSARIMA and Mixture of Experts forecaster)
in the [`advanced`](advanced) subdirectory. If you are interested in other utilities offered by the `merlion`
package, look at the resources inside the [`misc`](misc) subdirectory. For example,
[`misc/generate_synthetic_tsad_dataset.py`](misc/generate_synthetic_tsad_dataset.py)
is a script for generating an artifical anomaly detection dataset using `merlion`'s time series
generation and anomaly injection modules. This particular dataset may be loaded using the data
loader `ts_datasets.anomaly.Synthetic`.
