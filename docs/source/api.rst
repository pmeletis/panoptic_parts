API Reference
=============

We provide a public, stable API consisting of tested modules. However, in members of the API you may encounter experimental features (e.g. arguments or functions). These have the preffic `experimental\_` and are excempted from stability guarantees.

The functions of the API are exported (apart from their original modules) also in the panoptic_parts namespace, so they can be imported and used as:

.. code-block:: python

   import panoptic_parts as pp
   pp.decode_uids(uids)



Label format handling
---------------------

.. autofunction:: panoptic_parts.utils.format.decode_uids
.. autofunction:: panoptic_parts.utils.format.encode_ids

Visualization
-------------

.. autofunction:: panoptic_parts.utils.visualization.random_colors
.. autofunction:: panoptic_parts.utils.visualization.uid2color

Misc
----

.. autofunction:: panoptic_parts.utils.utils.safe_write