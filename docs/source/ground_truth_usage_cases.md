## Ground Truth usage cases

<!-- We provide for each image a single (image-like) ground truth file encoding semantic-, instance-, and parts- levels annotations. Our compact [label format](label_format.md) together with [_decode_uids_](api.html#panoptic_parts.utils.format.decode_uids) function enable easy decoding of the labels for various image understanding tasks including: -->

<!-- This is a workaround for internal reference to an API function problem using Markdown with Sphinx. -->
```eval_rst
We provide for each image a single (image-like) ground truth file encoding semantic-, instance-, and parts- levels annotations. Our compact :doc:`Label format <label_format>` together with 
:func:`panoptic_parts.utils.format.decode_uids`
function enable easy decoding of the labels for various image understanding tasks including:
```

```Python
# labels: Python int, or np.ndarray, or tf.Tensor, or torch.tensor

# Semantic Segmentation
semantic_ids, _, _ = decode_uids(labels)

# Instance Segmentation
semantic_ids, instance_ids, _ = decode_uids(labels)

# Panoptic Segmentation
_, _, _, semantic_instance_ids = decode_uids(labels, return_sids_iids=True)

# Parts Segmentation / Parts Parsing
_, _, _, semantic_parts_ids = decode_uids(labels, return_sids_pids=True)

# Instance-level Parts Parsing
semantic_ids, instance_ids, parts_ids = decode_uids(labels)

# Parts-level Panoptic Segmentation
_, _, _, semantic_instance_ids, semantic_parts_ids = decode_uids(labels, return_sids_iids=True, return_sids_pids=True)

```