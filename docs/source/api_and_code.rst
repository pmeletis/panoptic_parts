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


Code Reference
==============

Documented/Undocumented functionality of the rest of the code his repo lies here. This functionality will be added to the API in the future. Till then, the following functions may be moved or be unstable.

Dataset & Evaluation specifications
-----------------------------------

.. autoclass:: panoptic_parts.specs.dataset_spec.DatasetSpec
   :members:
   :undoc-members:

.. autoclass:: panoptic_parts.specs.eval_spec.PartPQEvalSpec
   :members:
   :undoc-members:

.. autoclass:: panoptic_parts.specs.eval_spec.SegmentationPartsEvalSpec
   :members:
   :undoc-members:

Visualization
-------------

.. autofunction:: panoptic_parts.visualization.visualize_label_with_legend.visualize_from_paths
.. autofunction:: panoptic_parts.utils.visualization.experimental_colorize_label
.. autofunction:: panoptic_parts.utils.visualization._generate_shades
.. autofunction:: panoptic_parts.utils.visualization._num_instances_per_sid
.. autofunction:: panoptic_parts.utils.visualization._num_parts_per_sid
.. autofunction:: panoptic_parts.utils.visualization._sid2iids
.. autofunction:: panoptic_parts.utils.visualization._sid2pids

Evaluation
----------

.. autofunction:: panoptic_parts.utils.evaluation_PartPQ.evaluate_PartPQ_multicore
.. autoclass::  panoptic_parts.utils.experimental_evaluation_IOU.ConfusionMatrixEvaluator_v2
   :members:
   :undoc-members:
   :show-inheritance:

Misc
----

.. autofunction:: panoptic_parts.utils.utils.compare_pixelwise
.. autofunction:: panoptic_parts.utils.utils._sparse_ids_mapping_to_dense_ids_mapping