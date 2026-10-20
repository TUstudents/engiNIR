Visualization
=============

All plot functions in :mod:`enginir.visualization` return a
``matplotlib.axes.Axes`` object so they can be embedded in larger figure
layouts or further customised with standard matplotlib calls.

Every function accepts an optional ``ax`` keyword argument.  Pass an existing
axes to draw into a subplot or an externally-created figure.

Plot inputs are validated before matplotlib is called. Spectral matrices,
score matrices, maps, and paired vectors must be numeric, finite (``y_map`` may
contain ``NaN`` for masked pixels), non-empty, and have matching axis lengths.

Spectra
-------

.. code-block:: python

   from enginir.visualization import plot_spectra

   # plain stack plot
   plot_spectra(X, wavelengths=wl)

   # color each spectrum by a property value (e.g. moisture content)
   plot_spectra(X, wl, color_by=y, title="Colored by moisture")

   # annotate individual spectra by name (shown for <= 20 samples)
   plot_spectra(X, wl, labels=sample_names)

Reference vs Predicted
-----------------------

.. code-block:: python

   from enginir.visualization import plot_prediction

   plot_prediction(y_test, y_pred, units="%", title="Test set")

The plot shows a scatter of reference vs predicted values, a dashed 1:1
reference line, and an :math:`R^2` annotation.

Loadings
--------

.. code-block:: python

   from enginir.visualization import plot_loadings

   # first three components by default
   plot_loadings(model.x_loadings_, wl)

   # select specific components (0-based)
   plot_loadings(model.x_loadings_, wl, components=[0, 1, 3])

Scores
------

.. code-block:: python

   from enginir.visualization import plot_scores

   T = model.transform(X)
   plot_scores(T, y=y, title="PLS scores colored by response")

   # label individual samples (shown for <= 30 samples)
   plot_scores(T, labels=sample_names, pc_x=0, pc_y=1)

VIP Scores
----------

.. code-block:: python

   from enginir.visualization import plot_vip

   vip = model.vip_scores()
   plot_vip(vip, wl, threshold=1.0)

Bars above the threshold (default 1.0) are colored distinctly.

Leverage / Outlier Map
----------------------

.. code-block:: python

   from enginir.visualization import plot_leverage
   from enginir.validation.outliers import leverage_outliers, spectral_outliers

   T = model.transform(X)
   lev, _ = leverage_outliers(T)
   X_recon = (T @ model.x_loadings_.T) * model.x_std_ + model.x_mean_
   q, q_crit, _ = spectral_outliers(X, X_recon)

   plot_leverage(
       lev, q,
       leverage_crit=3 * (T.shape[1] + 1) / T.shape[0],
       q_crit=q_crit,
       labels=sample_names,
   )

Samples outside either critical boundary are highlighted in red and
annotated with their label.

Hyperspectral Image Map
-----------------------

.. code-block:: python

   from enginir.visualization import plot_image_map
   from enginir.utils import flatten_cube, unflatten_cube

   # cube: (frames, rows, n_wl), already in the signal unit the model expects
   X_2d, spatial_shape = flatten_cube(cube)
   y_map = unflatten_cube(model.predict(X_2d), spatial_shape)   # (frames, rows)

   plot_image_map(y_map, title="Predicted moisture")

   # with an RGB background built from three bands
   rgb = cube[:, :, [50, 30, 10]]
   plot_image_map(y_map, background_rgb=rgb, cmap="plasma")

Pixels that are ``NaN`` in ``y_map`` (e.g. masked background) are left
transparent.

Composing subplots
------------------

Pass an ``ax`` argument to draw into a specific subplot:

.. code-block:: python

   import matplotlib.pyplot as plt
   from enginir.visualization import plot_spectra, plot_prediction

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
   plot_spectra(X, wl, ax=ax1, title="Preprocessed spectra")
   plot_prediction(y_test, y_pred, units="%", ax=ax2)
   fig.tight_layout()
   plt.savefig("diagnostics.png", dpi=150)
