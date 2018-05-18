.. currentmodule:: pyplanscoring

Installing
==========

You can get :obj:`pyplanscoring` from its current source on GitHub, to get all
the latest and greatest features. :obj:`pyplanscoring` is under active development,
and many new features are being added. However, note that the API is currently
unstable at this time.

.. code-block:: bash

   git clone https://github.com/victorgabr/pyplanscoring.git
   cd ./pyplanscoring
   python setup.py install

Requirements
============

PyPlanscoring as was designed to be a modern python library.
It targets long-term support using the newest features in data science.
It is recommended to set up a python 3.6 or higher environment because typing annotations will be adopted gradually.

Installing python using `Anaconda/Miniconda <https://conda.io/miniconda.html>`_ environment and conda-forge channel is highly recommended.


.. code-block:: bash

   python>=3.6.0
   pillow>=5.1.0
   pydicom>=1.0.2
   numba>=0.37.0
   numpy>=1.12.1
   scipy>=1.0.0
   pandas>=0.22.0
   quantities>=0.12.1

Optional
--------
These packages have to be installed to activate visualization and multiprocessing
capabilities.

.. code-block:: bash

   matplotlib>=2.0.0
   joblib>=0.11
   vispy>=0.5.3

conda environment
-----------------
It is possible to install all dependencies using:

.. code-block:: bash

  conda install -c conda-forge pillow pydicom numba numpy scipy pandas quantities matplotlib joblib vispy

After installing `Anaconda/Miniconda <https://conda.io/miniconda.html>`_ Python 3.6.

GPU computing
-------------
There is an experimental support to GPU computing :obj:`pyplanscoring.core.gpu_code`. DVH calculation kernels were written in `Numba <https://numba.pydata.org/numba-doc/dev/cuda/overview.html>`_.
It supports CUDA GPU programming by directly compiling a restricted subset of Python code into CUDA kernels and device functions following the CUDA execution model.

.. code-block:: bash

  conda install cudatoolkit
