enginir.models
==============

Regression
----------

.. autoclass:: enginir.models.PLSRegression
   :members: fit, predict, transform, vip_scores

.. autoclass:: enginir.models.PCR
   :members: fit, predict, transform

.. autoclass:: enginir.models.SVRModel
   :members: fit, predict

.. autoclass:: enginir.models.RandomForestModel
   :members: fit, predict, score, feature_importances_

.. autoclass:: enginir.models.MLPModel
   :members: fit, predict, score

Classification
--------------

.. autoclass:: enginir.models.SIMCA
   :members: fit, predict, q_residuals

.. autofunction:: enginir.models.classification.jackson_mudholkar_limit

.. autoclass:: enginir.models.LDAClassifier
   :members: fit, predict, predict_proba, transform
