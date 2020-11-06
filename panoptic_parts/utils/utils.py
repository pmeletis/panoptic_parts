import os
import os.path as op
import random

import numpy as np
from PIL import Image

from panoptic_parts.utils.format import decode_uids, encode_ids

# Functions that start with underscore (_) should be considered as internal.
# All other functions belong to the public API.
# Arguments and functions defined with the preffix experimental_ may be changed
# and are not backward-compatible.

# PUBLIC_API = [safe_write]


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
    False is path exists. True if the `image` is successfully written.
  """
  if op.exists(path):
    print('File already exists:', path)
    return False

  os.makedirs(op.dirname(path), exist_ok=True)
  Image.fromarray(image).save(path)
  return True

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

def color_map(N=256, normalized=False):
  """ 
  Python implementation of the color map function for the PASCAL VOC data set. 
  Official Matlab version can be found in the PASCAL VOC devkit 
  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
  """
  def bitget(byteval, idx):
      return ((byteval & (1 << idx)) != 0)

  dtype = 'float32' if normalized else 'uint8'
  cmap = np.zeros((N, 3), dtype=dtype)
  for i in range(N):
    r = g = b = 0
    c = i
    for j in range(8):
      r = r | (bitget(c, 0) << 7-j)
      g = g | (bitget(c, 1) << 7-j)
      b = b | (bitget(c, 2) << 7-j)
      c = c >> 3

    cmap[i] = np.array([r, g, b])

  cmap = cmap/255 if normalized else cmap
  return cmap


def _parse_sid_group(sid, sid_group):
  assert sid >= 1
  # import itertools
  # pids_all = itertools.chain.from_iterable(sid_group)
  sid_pid_old2sid_pid_new = dict()
  for i, pids_group in enumerate(sid_group, start=1):
    for pid in pids_group:
      sid_pid_old2sid_pid_new[sid * 100 + pid] = sid * 100 + i
  return sid_pid_old2sid_pid_new


def _transform_uids(uids, max_sid, sid2pids_groups):
  sid_pid_old2sid_pid_new = dict()
  for sid, sid_group in sid2pids_groups.items():
    sid_pid_old2sid_pid_new.update(_parse_sid_group(sid, sid_group))
  sid_pid_old2sid_pid_new[0] = 0
  for sid in range(1, max_sid + 1):
    if sid not in sid_pid_old2sid_pid_new.keys():
      sid_pid_old2sid_pid_new[sid] = sid
    for pid in range(100):
      sid_pid = sid * 100 + pid
      if pid == 0:
        sid_pid_old2sid_pid_new[sid_pid] = sid
        continue
      if sid_pid not in sid_pid_old2sid_pid_new.keys():
        sid_pid_old2sid_pid_new[sid_pid] = sid
  sid_pid_old2sid_pid_new = dict(sorted(sid_pid_old2sid_pid_new.items()))
  palette = np.asarray(list(sid_pid_old2sid_pid_new.values()), dtype=np.int32)

  _, iids, _, sids_pids = decode_uids(uids, return_sids_pids=True)
  sids_pids = palette[sids_pids]
  sids = np.where(sids_pids <= 99, sids_pids, sids_pids // 100)
  pids = np.where(sids_pids <= 99, -1, sids_pids % 100)
  return encode_ids(sids, iids, pids)
