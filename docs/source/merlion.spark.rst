merlion.spark package
=====================
This module implements APIs to integrate Merlion with PySpark. The expected use case is to
use distributed computing to train and run inference on multiple time series in parallel.

There are two ways to use the PySpark API: directly invoking the Spark apps
``spark_apps/anomaly.py`` and ``spark_apps/forecast.py`` from the command line with ``spark-submit``,
or using the Dockerfile to serve a Spark application on a Kubernetes cluster with ``spark-on-k8s``.
To understand the expected arguments for these apps, call ``spark_apps/anomaly.py -h`` or ``spark_apps/forecast.py -h``.

Setting up the spark-on-k8s-operator
------------------------------------
We will now cover how to serve these Spark apps using the
`spark-on-k8s-operator <https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/>`__.
For all methods, we expect that you have installed Merlion from source by cloning our
`git repo <https://github.com/salesforce/Merlion>`__.

Next, you need to create a Kubernetes cluster.
For local development, we recommend `Minikube <https://minikube.sigs.k8s.io/docs/start/>`__.
However, you can also use Kubernetes clusters managed by major cloud providers, e.g.
`Google's GKE <https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/blob/master/docs/gcp.md>`__ or
`Amazon's EKS <https://github.com/aws-samples/amazon-eks-apache-spark-etl-sample>`__. Setting up these clusters
is beyond the scope of this document, so we defer to the linked resources.

Once your Kubernetes cluster is set up, you need to use `Helm <https://helm.sh/docs/intro/install/>`__ to install
the ``spark-on-k8s-operator``. A full quick start guide for the operator can be found
`here <https://github.com/GoogleCloudPlatform/spark-on-k8s-operator/blob/master/docs/quick-start-guide.md>`__,
but the key steps are to call

.. code-block:: shell

   $ helm repo add spark-operator https://googlecloudplatform.github.io/spark-on-k8s-operator
   $ kubectl create namespace spark-apps
   $ helm install spark-operator spark-operator/spark-operator \
     --namespace spark-operator --create-namespace --set sparkJobNamespace=spark-apps

This will create a Kubernetes namespace ``spark-apps`` from which all your Spark applications will run, and it will
use Helm to install the ``spark-on-k8s-operator`` (which manages all running PySpark apps as Kubernetes custom
resources) in the namespace ``spark-operator``.

Then, you can build the provided Dockerfile with ``docker build -t merlion-spark -f docker/spark-on-k8s/Dockerfile .``
from the root directory of Merlion.
If you are using Minikube, make sure to point your shell to Minikube's Docker daemon with
``eval $(minikube -p minikube docker-env)`` before building the image.
If you are working on the cloud, you will need to publish the built Docker image to the appropriate registry, e.g.
`Google's gcr.io <https://cloud.google.com/container-registry/>`__ or `Amazon's ECR <https://aws.amazon.com/ecr/>`__.

Specifying a Spark App
----------------------
Once your cluster is set up, you can submit a YAML file specifying your spark application as a Kubernetes custom
resource. We provide templates for both forecasting and anomaly detection in ``k8s-spec/forecast.yml`` and
``k8s-spec/anomaly.yml`` respectively. Both of these use the ``walmart_mini.csv`` dataset,
which contains the weekly sales of 10 different products at 2 different stores.

API Documentation
-----------------
The API documentation of Merlion's PySpark connectors (``merlion.spark``) is below.

.. automodule:: merlion.spark
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
    dataset
    pandas_udf

Submodules
----------

merlion.spark.dataset module
----------------------------

.. automodule:: merlion.spark.dataset
   :members:
   :undoc-members:
   :show-inheritance:

merlion.spark.pandas\_udf module
--------------------------------

.. automodule:: merlion.spark.pandas_udf
   :members:
   :undoc-members:
   :show-inheritance:

