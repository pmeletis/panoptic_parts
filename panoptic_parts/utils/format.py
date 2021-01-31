"""
Utility functions for reading and writing to our hierarchical panoptic format (see README).
Tensorflow and Pytorch are optional frameworks.
"""

from enum import Enum
import functools

import numpy as np

TENSORFLOW_IMPORTED = False
try:
  import tensorflow as tf # pylint: disable=import-error
  TENSORFLOW_IMPORTED = True
except ModuleNotFoundError:
  pass

TORCH_IMPORTED = False
try:
  import torch # pylint: disable=import-error
  TORCH_IMPORTED = True
except ModuleNotFoundError:
  pass

# Functions that start with underscore (_) should be considered as internal.
# All other functions belong to the public API.
# Arguments and functions defined with the preffix experimental_ may be changed
# and are not backward-compatible.

# PUBLIC_API = [decode_uids, encode_ids]

class Frameworks(Enum):
  PYTHON = "python"
  NUMPY = "numpy"
  TENSORFLOW = "tensorflow"
  TORCH = "torch"

def _validate_uids_values_numpy_python(uids):
  assert isinstance(uids, (int, np.int32, np.ndarray))
  uids = np.array(uids, dtype=np.int32)
  if np.any(uids > 99_999_99):
    raise ValueError('Some uids exceed the 99_999_99 encoding limit.')
  if np.any(uids < 0):
    raise ValueError('Some uids are negative.')
  # TODO(panos): this gives a warning for 0 uids
  num_digits = (np.log10(uids) + 1).astype(np.int)
  if 3 in np.unique(num_digits):
    raise ValueError(
        'Some uids have length of 3 digits that is not allowed by the encoding format.')


def _decode_uids_functors_and_checking(uids):
  # this functions makes the decode_uids more clear
  # required frameworks: Python and NumPy
  # TODO(panos): split this function into 2
  if isinstance(uids, np.ndarray):
    if uids.dtype != np.int32:
      raise TypeError(f'{uids.dtype} is an unsupported dtype of np.ndarray uids.')
    where = np.where
    ones_like = np.ones_like
    divmod_ = np.divmod
    maximum = np.maximum
    dtype = np.int32
    _validate_uids_values_numpy_python(uids)
    return where, ones_like, divmod_, maximum, dtype
  if isinstance(uids, (int, np.int32)):
    where = lambda cond, true_br, false_br: true_br if cond else false_br
    ones_like = lambda the_like: type(the_like)(1)
    divmod_ = divmod
    maximum = max
    dtype = (lambda x: x) if isinstance(uids, int) else np.int32
    _validate_uids_values_numpy_python(uids)
    return where, ones_like, divmod_, maximum, dtype

  # optional frameworks: Tensorflow and Pytorch
  if TENSORFLOW_IMPORTED:
    if isinstance(uids, tf.Tensor):
      if uids.dtype != tf.int32:
        raise TypeError(f'{uids.dtype} is an unsupported dtype of tf.Tensor uids.')
      where = tf.where
      ones_like = tf.ones_like
      divmod_ = lambda x, y: (x // y, x % y)
      maximum = tf.maximum
      dtype = lambda x: x
      return where, ones_like, divmod_, maximum, dtype
  if TORCH_IMPORTED:
    if isinstance(uids, torch.Tensor):
      where = torch.where
      ones_like = torch.ones_like
      divmod_ = lambda x, y: (x // y, x % y)
      maximum = torch.max
      dtype = functools.partial(torch.tensor, dtype=torch.int32)
      return where, ones_like, divmod_, maximum, dtype

  raise TypeError(f'{type(uids)} is an unsupported type of uids.')

def decode_uids(uids, return_sids_iids=False, return_sids_pids=False):
  """
  Given the universal ids `uids` according to the hierarchical format described
  in README, this function returns element-wise
  the semantic ids (sids), instance ids (iids), and part ids (pids).
  Optionally it returns the sids_iids and sids_pids as well.

  sids_iids represent the semantic-instance-level (two-level) labeling,
  e.g., sids_iids(Cityscapes-Panoptic-Parts) = Cityscapes-Original.

  sids_pids = sids * 100 + max(pids, 0) and represent the semantic-part-level labeling,
  e.g., sids_pids(23_003_04) = 23_04.

  Examples (output is same type as input - not shown for clarity):
    - decode_uids(23) → (23, -1, -1)
    - decode_uids(23003) → (23, 3, -1)
    - decode_uids(2300304) → (23, 3, 4)
    - decode_uids(tf.constant([1, 12, 1234, 12345, 123456, 1234567])) →
      ([ 1, 12,   1,  12,   1,  12],
       [-1, -1, 234, 345, 234, 345],
       [-1, -1,  -1,  -1,  56,  67])
    - decode_uids(np.array([[1, 12], [1234, 12345]])) →
      ([[ 1, 12], [ 1, 12]],
       [[ -1,  -1], [234, 345]],
       [[-1, -1], [-1, -1]])

  Args:
    uids: tf.Tensor of dtype tf.int32 and arbitrary shape,
          or np.ndarray of dtype np.int32 and arbitrary shape,
          or torch.tensor of dtype torch.int32 and arbitrary shape,
          or Python int,
          or np.int32 integer,
          with elements according to hierarchical format (see README).

  Return:
    sids, iids, pids: same type and shape as uids, with -1 for not relevant pixels.
  
    sids will have no -1.
    iids will have -1 for pixels labeled with sids only.
    pids will have -1 for pixels labeled with sids or sids_iids only.
  
    if return_sids_iids:
      sids_iids: same type and shape as uids, will have no -1.
    if return_sids_pids:
      sids_pids: same type and shape as uids, will have no -1.
  """
  where, ones_like, divmod_, maximum, dtype = _decode_uids_functors_and_checking(uids)

  # explanation for using dtype and np.asarray in this function:
  #   dtype: numpy implicitly converts Python int literals in np.int64, we need np.int32
  #   np.asarray: numpy implicitly converts ndarray with one element to np.int32 (which is not
  #     ndarray), moreover dtypes are implicitly converted to np.int64 for arithmetic operations

  # pad uids to uniform 7-digit length
  uids_padded = where(uids <= 99_999,
                      where(uids <= 99, uids * dtype(10**5), uids * dtype(10**2)),
                      uids)
  # split uids to components (sids, iids, pids) from right to left
  sids_iids, pids = divmod_(uids_padded, dtype(10**2))
  sids, iids = divmod_(sids_iids, dtype(10**3))
  invalid_ids = ones_like(uids) * dtype(-1)
  # set invalid ids
  iids = where(uids <= 99, invalid_ids, iids)
  pids = where(uids <= 99_999, invalid_ids, pids)

  if isinstance(uids, np.ndarray):
    sids = np.asarray(sids, dtype=np.int32)
  returns = (sids, iids, pids)

  if return_sids_iids:
    sids_iids = where(uids <= 99_999, uids, sids_iids)
    returns += (sids_iids,)

  if return_sids_pids:
    sids_pids = where(uids <= 99_999, sids, sids * dtype(10**2) + maximum(pids, dtype(0)))
    if isinstance(uids, np.ndarray):
      sids_pids = np.asarray(sids_pids, dtype=np.int32)
    returns += (sids_pids,)

  return returns


def _validate_ids_values_numpy_python(sids, iids, pids):
  assert isinstance(sids, (int, np.int32, np.ndarray))
  assert type(sids) is type(iids) and type(iids) is type(pids)
  sids, iids, pids = map(functools.partial(np.array, dtype=np.int32), [sids, iids, pids])
  if np.any(sids > 99):
    raise ValueError('Some sids exceed the 99 encoding limit.')
  if np.any(iids > 999):
    raise ValueError('Some iids exceed the 999 encoding limit.')
  if np.any(pids > 99):
    raise ValueError('Some sids exceed the 99 encoding limit.')
  if np.any(sids < -1):
    raise ValueError('Some sids are negative.')
  if np.any(iids < -1):
    raise ValueError('Some iids are negative.')
  if np.any(pids < -1):
    raise ValueError('Some pids are negative.')


def _encode_ids_functors_and_checking(sids, iids, pids):
  # this functions makes the endoce_ids more clear
  # required frameworks: Python and NumPy
  if type(sids) is not type(iids) and type(iids) is not type(pids):
    raise ValueError(
        f"All arguments must have the same type, given {(*map(type, (sids, iids, pids)),)}")
  if isinstance(sids, np.ndarray):
    if sids.dtype != np.int32:
      raise TypeError(f'{sids.dtype} is an unsupported dtype of np.ndarray ids.')
    where = np.where
    _validate_ids_values_numpy_python(sids, iids, pids)
    return where
  if isinstance(sids, (int, np.int32)):
    where = lambda cond, true_br, false_br: true_br if cond else false_br
    _validate_ids_values_numpy_python(sids, iids, pids)
    return where

  # optional frameworks: Tensorflow and Pytorch
  if TENSORFLOW_IMPORTED:
    if isinstance(sids, tf.Tensor):
      if sids.dtype != tf.int32:
        raise TypeError(f'{sids.dtype} is an unsupported dtype of tf.Tensor ids.')
      where = tf.where
      return where
  if TORCH_IMPORTED:
    if isinstance(sids, torch.Tensor):
      where = torch.where
      return where

  raise TypeError(f'{type(sids)} is an unsupported type of ids.')

def encode_ids(sids, iids, pids):
  """
  Given semantic ids (sids), instance ids (iids), and part ids (pids)
  this function encodes them element-wise to uids
  according to the hierarchical format described in README.

  This function is the opposite of decode_uids, i.e.,
  uids = encode_ids(decode_uids(uids)).

  Args:
    sids, iids, pids: all of the same type with -1 for non-relevant pixels with
      elements according to hierarchical format (see README). Can be:
      tf.Tensor of dtype tf.int32 and arbitrary shape,
      or np.ndarray of dtype np.int32 and arbitrary shape,
      or torch.tensor of dtype torch.int32 and arbitrary shape,
      or Python int,
      or np.int32 integer.

  Return:
    uids: same type and shape as the args according to hierarchical format (see README).
  """
  # TODO!!!(panos): encode_ids(sid, -1, pid) ends with a sid_pid and not a uid,
  #   is this behavior desirable?
  where = _encode_ids_functors_and_checking(sids, iids, pids)

  sids_iids = where(iids < 0, sids, sids * 10**3 + iids)
  uids = where(pids < 0, sids_iids, sids_iids * 10**2 + pids)

  return uids
