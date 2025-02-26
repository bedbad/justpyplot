justpyplot documentation
=======================

A fast, lightweight plotting library for real-time visualization.

Basic Usage
----------

.. code-block:: python

   import numpy as np
   from justpyplot import justpyplot as jplt

   x = np.linspace(0, 10, 50)
   y = np.sin(x)

   # Create plot components
   figure, grid, labels, title = jplt.plot(
       np.array([x, y]),
       title='Sine Wave'
   )

   # Blend components
   final_image = jplt.blend(grid, figure, labels, title)

Installation
-----------

.. code-block:: bash

   pip install justpyplot

API Reference
------------

Main Functions
~~~~~~~~~~~~~

.. automodule:: justpyplot.justpyplot
   :members:
   :imported-members: False
   :special-members: False
   :private-members: False
   :undoc-members: False

Text Rendering
~~~~~~~~~~~~~

.. automodule:: justpyplot.textrender
   :members:
   :imported-members: False
   :special-members: False
   :private-members: False
   :undoc-members: False

Indices
=======

* :ref:`genindex` 