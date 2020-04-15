
import os
import os.path as op
import random

import numpy as np
from PIL import Image

from utils.format import decode_uids

# Functions that start with underscore (_) should be considered as internal.
# All other functions belong to the public API.
# Arguments and functions defined with the preffix experimental_ may be changed
# and are not backward-compatible.

# PUBLIC_API = []

def _random_colors(num):
  """
  Return num random RGB colors in [0, 255] as a list of lists.
  Colors can be repeated, which is a preferred behavior so we don't run out of colors.
  """
  return [list(color) for color in np.random.choice(256, size=(num, 3))]

def _sparse_ids_mapping_to_dense_ids_mapping(ids_dict, void, length=None, dtype=np.int32):
  """
  Create a dense np.array from an ids dictionary. The array can be used
  for indexing, e.g. numpy advanced indexing or tensorflow gather.
  This method is useful to transform a dictionary of uids to class mappings (e.g. {2600305: 3}),
  to a dense np.array that has in position 2600305 the value 3. This in turn can be used in
  gathering operations. The reason that the mapping is given in a dictionary is due to its
  sparseness, e.g. we may not want to hard-code an array with 2600305 elements in order to
  have the mapping for the 2600305th element.

  ids.values() and void must have the same shape and dtype.

  The length of the dense_mapping is infered from the maximum value of ids_dict.keys(). If you
  need a longer dense_mapping provide the length in `length`.

  Args:
    ids_dict: dictionary mapping ids to numbers (usually classes),
    void: the positions of the dense array that don't appear in ids_dict.keys()
      will be filled with the void value,
    length: the length of the dense mapping can be explicitly provided
    dtype: the dtype of the returned dense mapping
  """
  # TODO(panos): add args requirements checking, and refactor this code
  # TODO(panos): check the validity of +1 (useful only if key 0 exists?)

  void_np = np.array(void)
  length_mapping = length or np.max(list(ids_dict.keys())) + 1

  if np.array(void).ndim == 0:
    dense_mapping = np.full(length_mapping, void, dtype=dtype)
    for uid, cid in ids_dict.items():
      dense_mapping[uid] = cid
  elif void_np.ndim == 1:
    dense_mapping = np.full((length_mapping, void_np.shape[0]), void, dtype=dtype)
    for k, v in ids_dict.items():
      dense_mapping[k] = v
  else:
    raise NotImplementedError('Not yet implemented.')

  return dense_mapping

def safe_write(path, image):
  """
  Check if `path` exist and if it doesn't creates all needed intermediate-level directories
  and saves `image` to `path`.

  Args:
    path: a path passed to os.path.exists, os.makedirs and PIL.Image.save()
    image: a numpy image passed to PIL.Image.fromarray()

  Return:
    True is path exists else False.
  """
  if op.exists(path):
    print('File already exists:', path)
    return True

  os.makedirs(op.dirname(path), exist_ok=True)
  Image.fromarray(image).save(path)
  return False

def uids_lids2uids_cids(uids_with_lids, lids2cids):
  """
  Convert uids with semantic classes encoded as lids to uids with cids.
  This function is useful in the Cityscapes context, or other datasets
  where the lids and cids separation is used.
  """
  uids = uids_with_lids
  sids, _, _ = decode_uids(uids)
  uids_with_cids = np.where(
      uids <= 99,
      lids2cids[sids],
      np.where(uids <= 99_999,
               lids2cids[sids] * 10**3 + uids % 10**3,
               lids2cids[sids] * 10**5 + uids % 10**5))

  return uids_with_cids
