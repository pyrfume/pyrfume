Welcome to Pyrfume's documentation!
===================================

|Pyrfume logo|

.. _pyrfume-is-a-python-library-for-olfactory-psychophysics-research-see-notebooks-for-examples-of-use:

``pyrfume`` is a python library for olfactory psychophysics research. See "notebooks" for examples of use.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Examples:
~~~~~~~~~

.. _note-these-require-the-pyrfume-data-library-provided-separately:

Note: these require the Pyrfume data library, provided separately.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   # Get raw data from the Sigma Fragrance & Flavor Catalog
   from pyrfume import sigma_ff
   descriptors, data = sigma_ff.get_data()

   # Get a PubChem CID-indexed dataframe of the odorant and descriptor data from that catalog:
   import pyrfume
   sigma = pyrfume.load_data('sigma/sigma.csv')

`Website`_
~~~~~~~~~~

.. _Website: http://pyrfume.org

.. |Pyrfume logo| image:: https://avatars3.githubusercontent.com/u/34174393?s=200&v=4
pandoc 2.9.2

https://pandoc.org


.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
