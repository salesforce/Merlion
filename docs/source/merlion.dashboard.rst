merlion.dashboard package
=========================

This package includes a GUI dashboard app for Merlion, providing a convenient way to train
and test a time series forecasting or anomaly detection model supported in Merlion. To launch
the dashboard app, type the following command: ``python -m merlion.dashboard``.

It will launch a Dash app on http://localhost:8050/ by default. After opening the link, the app
will create a folder ``merlion`` in your home directory. This folder includes the datasets you want to
analyze or train a model with (in the ``data`` folder), and the trained models for time series
forecasting or anomaly detection (in the ``models`` folder).

The app has three tabs. The first one is called "file manager" in which you can upload your datasets
(the datasets will be stored in ``~/merlion/data``), check basic statistics of the datasets, visualize
the time series data, or download a particular trained model:

.. image:: _static/figures/dashboard_file.png

You can click "Drag & Drop" to upload the file to the ``merlion`` folder (our app is designed to support
docker deployment, so it doesn't allow to open a local file directly). If you use the app on a local
machine, you can also copy the data to ``~/merlion/data`` directly. The supported data file is in
the csv format, where the first column should be either integer Unix timestamps in milliseconds, or datetimes in a
string format (e.g., "1970-01-01 00:00:00"). The other columns are the features/variables.

Clicking the load button will load the dataset and show the time series figure on the right hand side.
It will also show some basic statistics, e.g., time series length, mean/std for each variable.
If you have already trained a model using the dashboard, you can select the model you want to download
and click the download button. The model and its configuration file will be compressed into a zip file.

The second tab is used to train a time series anomaly detection model:

.. image:: _static/figures/dashboard_anomaly.png

The app provides full support for these models, where you can choose different algorithms and set particular parameters
according to your needs. To train a model, you need to:

- **Select the dataset**: You can select a single training dataset if there is no test dataset, and then choose
  a train/test split fraction for splitting this dataset into training and test dataset for evaluation.
  If you have the test dataset, you can choose "Separate train/test files" and select the test dataset,
  and then the model will be trained with the training dataset and evaluated with the separate test dataset.
  The screenshot above uses a single data file, where we use the first 15% for training and the last 85% for testing.
- **Set the feature columns**: Merlion supports both univariate and multivariate time series anomaly detection,
  so you can choose one or more features on which to train an anomaly detection model.
- **Set the label column**: If the dataset has a label column, you can set it for evaluation. Otherwise,
  ignore this setting.
- **Select an anomaly detection algorithm**: You need to choose an anomaly detection algorithm such as
  IsolationForest. You may modify the model's hyperparameters if the default values do not work well.
- **Set threshold parameters**: You can also test different settings for the detection threshold to
  determine which value is better for your specific application. Note that updating the threshold will
  not re-train the entire model; it will simply change the post-processing applied by the trained model.

The training procedure begins after clicking the train button, and the trained model is saved in the
folder ``~/merlion/models/algorithm_name``. The figure on the right hand side shows the detection results
on the test dataset, and the tables show the training and testing performance metrics if you set the
label column.

The third tab is used to train a time series forecasting model supported in Merlion:

.. image:: _static/figures/dashboard_forecast.png

The app provides full support for these models, where you can choose different algorithms and set particular parameters
according to your needs. To train a model, you need to:

- **Select the dataset**: You can select a single training dataset if there is no test dataset, and then choose
  a train/test split fraction for splitting this dataset into training and test dataset for evaluation.
  If you have the test dataset, you can choose "Separate train/test files" and select the test dataset,
  and then the model will be trained with the training dataset and evaluated with the separate test dataset.
  The screenshot above uses separate train/test files.
- **Set the target column**: You need to set the target column whose value you wish to forecast (required),
  any additional features to use for `multivariate forecasting <tutorials/forecast/2_ForecastMultivariate>` (optional),
  and the `exogenous variables <tutorials/forecast/3_ForecastExogenous>` whose values are known a priori (optional).
- **Select a forecasting algorithm**: Finally, you need to choose a forecasting algorithm such as
  Arima, AutoETS. You may modify the model's hyperparameters if the default values do not work well.

The training procedure begins after clicking the train button. It may take some time to finish model
training. After the model is trained, the model files will be saved in the folder ``~/merlion/models/algorithm_name``.
The figure on the right hand side shows the forecasting results on the test dataset, and the tables
show the training and testing performance metrics.
