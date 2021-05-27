API Reference
=============

We provide a public, backwards compatible API, which allows easier bug fixes and functionality updates. We suggest that the users update their local clone of this repository frequently by pulling the master branch.

All functions and arguments named with the preffix 'experimental\_' or with an '_' do not
belong to the stable API and may change.

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