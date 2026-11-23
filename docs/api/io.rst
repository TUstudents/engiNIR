enginir.io
==========

.. automodule:: enginir.io
   :no-members:

Core API
--------

load_spectra
~~~~~~~~~~~~

.. autofunction:: enginir.io.load_spectra

SpectralData
~~~~~~~~~~~~

.. autoclass:: enginir.io.SpectralData
   :no-members:

file_format
~~~~~~~~~~~

.. autofunction:: enginir.io.file_format

EnginirIOError
~~~~~~~~~~~~~~

.. autoexception:: enginir.io.EnginirIOError

CSV / Excel
-----------

load_csv
~~~~~~~~

.. autofunction:: enginir.io.load_csv

save_csv
~~~~~~~~

.. autofunction:: enginir.io.save_csv

Galactic SPC
------------

load_spc
~~~~~~~~

.. autofunction:: enginir.io.load_spc

Thermo SPA
----------

load_spa
~~~~~~~~

.. autofunction:: enginir.io.load_spa

JCAMP-DX
--------

load_jcamp
~~~~~~~~~~

.. autofunction:: enginir.io.load_jcamp

ENVI
----

load_envi
~~~~~~~~~

.. autofunction:: enginir.io.load_envi

iter_envi_frame_chunks
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.iter_envi_frame_chunks

save_envi
~~~~~~~~~

.. autofunction:: enginir.io.save_envi

Simple HDF5
-----------

load_hdf5
~~~~~~~~~

.. autofunction:: enginir.io.load_hdf5

iter_hdf5_frame_chunks
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.iter_hdf5_frame_chunks

save_hdf5
~~~~~~~~~

.. autofunction:: enginir.io.save_hdf5

Pixel binning
-------------

.. automodule:: enginir.io._binning
   :no-members:

bin_blocks
~~~~~~~~~~

.. autofunction:: enginir.io.bin_blocks

bin_frame_chunks
~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.bin_frame_chunks

bin_rows
~~~~~~~~

.. autofunction:: enginir.io.bin_rows

BINNING_FACTORS
~~~~~~~~~~~~~~~

.. autodata:: enginir.io.BINNING_FACTORS
   :no-value:

Wavelength tables
-----------------

.. automodule:: enginir.io.wavelength_tables
   :no-members:

wavelength_tables
~~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.wavelength_tables.wavelength_tables

wavelength_table
~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.wavelength_tables.wavelength_table

load_wavelength_table
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.wavelength_tables.load_wavelength_table

is_band_numbers
~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.wavelength_tables.is_band_numbers

user_table_folder
~~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.wavelength_tables.user_table_folder

WavelengthTable
~~~~~~~~~~~~~~~

.. autoclass:: enginir.io.wavelength_tables.WavelengthTable
   :members:

Project files
-------------

.. automodule:: enginir.io.project_store
   :no-members:

Creating and loading
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.project_store.create_project

.. autofunction:: enginir.io.project_store.load_project_store

FILE_SCHEMA_VERSION
~~~~~~~~~~~~~~~~~~~

.. autodata:: enginir.io.FILE_SCHEMA_VERSION

Converting projects of engiNIR 1.5 and 1.6
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.migrate_project

Saving objects
~~~~~~~~~~~~~~

.. autofunction:: enginir.io.project_store.save_dataset_ref

.. autofunction:: enginir.io.project_store.save_reference_correction

.. autofunction:: enginir.io.project_store.save_target_definition

.. autofunction:: enginir.io.project_store.save_target_assignment

.. autofunction:: enginir.io.project_store.save_preprocessing_recipe

.. autofunction:: enginir.io.project_store.save_signal_domain_contract

.. autofunction:: enginir.io.project_store.save_training_roi

.. autofunction:: enginir.io.project_store.save_roi_sample

.. autofunction:: enginir.io.project_store.save_roi_job

.. autofunction:: enginir.io.project_store.save_sample_set

.. autofunction:: enginir.io.project_store.save_model_run

.. autofunction:: enginir.io.project_store.save_prediction_run

.. autofunction:: enginir.io.project_store.remove_roi_from_project

.. autofunction:: enginir.io.project_store.remove_target_assignment

.. autofunction:: enginir.io.project_store.load_roi_band_images


.. autofunction:: enginir.io.project_store.relocate_dataset_ref

.. autofunction:: enginir.io.project_store.set_dataset_wavelengths

Writing large predictions incrementally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.project_store.initialize_prediction_run_artifacts

.. autofunction:: enginir.io.project_store.open_prediction_run_artifact_writer

.. autoclass:: enginir.io.project_store.PredictionArtifactWriter
   :members: write_chunk

.. autofunction:: enginir.io.project_store.finalize_prediction_run_artifacts

.. autofunction:: enginir.io.project_store.discard_prediction_run_artifacts

.. autofunction:: enginir.io.project_store.discard_staged_predictions

.. autofunction:: enginir.io.project_store.has_prediction_run

Loading arrays on demand
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.project_store.load_roi_spectra

.. autofunction:: enginir.io.project_store.load_roi_spectra_batch

.. autofunction:: enginir.io.project_store.load_roi_spectra_dtypes

.. autofunction:: enginir.io.project_store.load_roi_spectra_matrix

.. autofunction:: enginir.io.project_store.iter_roi_reference_spectra

.. autofunction:: enginir.io.project_store.load_model_run_artifact_refs

.. autofunction:: enginir.io.project_store.load_model_run_prediction_artifact

.. autofunction:: enginir.io.project_store.load_prediction_run_artifact_refs

.. autofunction:: enginir.io.project_store.load_prediction_spatial_map_preview

EVK EIFF
--------

load_eiff
~~~~~~~~~

.. autofunction:: enginir.io.load_eiff

iter_eiff_frame_chunks
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: enginir.io.iter_eiff_frame_chunks

Exceptions
----------

EIFFError
~~~~~~~~~

.. autoexception:: enginir.io.EIFFError

EIFFTruncatedError
~~~~~~~~~~~~~~~~~~

.. autoexception:: enginir.io.EIFFTruncatedError

EIFFMemoryError
~~~~~~~~~~~~~~~

.. autoexception:: enginir.io.EIFFMemoryError

SPAError
~~~~~~~~

.. autoexception:: enginir.io.SPAError

SPCError
~~~~~~~~

.. autoexception:: enginir.io.SPCError

JCAMPError
~~~~~~~~~~

.. autoexception:: enginir.io.JCAMPError

ENVIError
~~~~~~~~~

.. autoexception:: enginir.io.ENVIError
