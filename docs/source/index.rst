Welcome to Calibur🗡️!
===================================

**Calibur** is a sword against the many inconsistent and ambiguous conventions
(viewport origins, depth range, camera coordinate systems, etc.) in computer graphics and vision.

It is a library that translates between these conventions for you.
It also hosts as a library of documentation of different conventions and space definitions in CG and CV
that would otherwise be hard to find.

Installation
============

The calibur package can be installed by:

.. code-block::

   pip install calibur

Development
===========

Calibur is designed to be light-weight, and only depends on `numpy` and `typing_extensions`.
However, to run tests, more dependencies are required. To install them:

.. code-block::

   pip install coverage trimesh opencv-contrib-python

API Reference
=============
.. toctree::
   :maxdepth: 1
   
   API Reference <generated/calibur>
