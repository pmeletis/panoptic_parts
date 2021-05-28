import sys
assert float(sys.version[:3]) >= 3.7, 'This test uses Python >= 3.7 functionality.'
import functools
import copy
from typing import Dict
from dataclasses import dataclass
import glob

import numpy as np
import tensorflow as tf
assert tf.version.VERSION[0] == '2', 'This test uses TF r2.x functionality.'
import torch
assert torch.version.__version__[0] == '1', 'This test uses PyTorch r1.x functionality.'
from PIL import Image

from panoptic_parts.utils.format import decode_uids, encode_ids
from panoptic_parts.specs.dataset_spec import DatasetSpec

# TOTAL_TESTS = 128 + 30 = 158 in this file.

def encode_ids_cases():
  examples = [[(1, 2, 3), (1_002_03,)],
              [(11, 2, 3), (11_002_03,)],
              [(1, 2, -1), (1_002,)],
              [(11, 2, -1), (11_002,)],
              [(1, -1, -1), (1,)],
              [(11, -1, -1), (11,)],
  ]

  types_fn = [functools.partial(tf.constant, dtype=tf.int32),
              functools.partial(np.array, dtype=np.int32),
              int,
              np.int32,
              functools.partial(torch.tensor, dtype=torch.int32)]

  # cases: [[inputs, outputs], ...]
  cases = list()
  for example in examples:
    for type_fn in types_fn:
      inputs = [*map(type_fn, example[0])]
      outputs = [*map(type_fn, example[1])]
      cases.append([inputs, outputs])
  
  return cases

def encode_ids_test(cases):
  for case in cases:
    results = encode_ids(*case[0])
    assert results == case[1][0], (case, results)
  print(f"encode_ids: {len(cases)} test cases completed successfully.")

def decode_uids_cases():
  ersiT = {'return_sids_iids': True}
  erspT = {'return_sids_pids': True}
  ersipT = {'return_sids_iids': True, 'return_sids_pids': True}
  # examples = [example, ...]
  # example = [inputs, outputs] = [[args, kwargs], outputs]
  # examples = [ [ [args, kwargs], outputs ], ... ]
  examples = [
      [[(1,), dict()], (1, -1, -1)],
      [[(1,), ersiT], (1, -1, -1, 1)],
      [[(1,), erspT], (1, -1, -1, 1)],
      [[(1,), ersipT], (1, -1, -1, 1, 1)],
      [[(11,), dict()], (11, -1, -1)],
      [[(11,), ersiT], (11, -1, -1, 11)],
      [[(11,), erspT], (11, -1, -1, 11)],
      [[(11,), ersipT], (11, -1, -1, 11, 11)],
      [[(1_002,), dict()], (1, 2, -1)],
      [[(1_002,), ersiT], (1, 2, -1, 1002)],
      [[(1_002,), erspT], (1, 2, -1, 1)],
      [[(1_002,), ersipT], (1, 2, -1, 1002, 1)],
      [[(11_002,), dict()], (11, 2, -1)],
      [[(11_002,), ersiT], (11, 2, -1, 11002)],
      [[(11_002,), erspT], (11, 2, -1, 11)],
      [[(11_002,), ersipT], (11, 2, -1, 11002, 11)],
      [[(1_002_03,), dict()], (1, 2, 3)],
      [[(1_002_03,), ersiT], (1, 2, 3, 1002)],
      [[(1_002_03,), erspT], (1, 2, 3, 103)],
      [[(1_002_03,), ersipT], (1, 2, 3, 1002, 103)],
      [[(11_002_03,), dict()], (11, 2, 3)],
      [[(11_002_03,), ersiT], (11, 2, 3, 11002)],
      [[(11_002_03,), erspT], (11, 2, 3, 1103)],
      [[(11_002_03,), ersipT], (11, 2, 3, 11002, 1103)],
  ]

  types_fn = [
      functools.partial(tf.constant, dtype=tf.int32),
      # probably np.array(..., dtype=int32) does not enforce dtype to be np.int32
      # lambda x: np.array(x).astype(np.int32),
      functools.partial(np.array, dtype=np.int32),
      int,
      np.int32,
      functools.partial(torch.tensor, dtype=torch.int32)
  ]

  class DSpec:
    # this is needed because deepcopy does not copy class-level attributes
    def __init__(self):
      self._sid_pid_file2sid_pid: Dict = {1_01: 1_02, 11_02: 11_01}
      self.sid2scene_class = {k: str(k) if k != 0 else 'UNLABELED' for k in range(12)}
      self.sid_pid2scene_class_part_class = {k: (v, 'UNLABELED') for k, v in self.sid2scene_class.items()}
      self.sid_pid2scene_class_part_class.update({1_01: ('1', '1'), 1_02: ('1', '2'),
                                                  11_01: ('11', '1'), 11_02: ('11', '2')})

  dataset_spec1 = DSpec()
  ersipTdc1 = copy.copy(ersipT)
  ersipTdc1['experimental_dataset_spec'] = dataset_spec1
  ersipTdc1['experimental_correct_range'] = True
  dataset_spec2 = copy.deepcopy(dataset_spec1)
  del dataset_spec2._sid_pid_file2sid_pid
  ersipTdc2 = copy.copy(ersipT)
  ersipTdc2['experimental_dataset_spec'] = dataset_spec2
  ersipTdc2['experimental_correct_range'] = True
  more_examples = [
      [[(1_002_01,), ersipT], (1, 2, 1, 1002, 1_01)],
      [[(1_002_01,), ersipTdc1], (1, 2, 2, 1002, 1_02)],
      [[(11_002_02,), ersipT], (11, 2, 2, 11002, 11_02)],
      [[(11_002_02,), ersipTdc1], (11, 2, 1, 11002, 11_01)],
      [[(12,), ersipTdc2], (0, -1, -1, 0, 0)],
      [[(12_002_01,), ersipTdc2], (0, -1, -1, 0, 0)],
      [[(12_002_04,), ersipTdc2], (0, -1, -1, 0, 0)],
      [[(11_002_03,), ersipTdc2], (11, 2, -1, 11_002, 11)],
  ]

  # cases: [[[args, kwargs], outputs], ...]
  cases = list()
  for example in examples:
    for type_fn in types_fn:
      inputs = [(type_fn(example[0][0][0]),), example[0][1]]
      outputs = (*map(type_fn, example[1]),)
      cases.append([inputs, outputs])

  type_nparray = functools.partial(np.array, dtype=np.int32)
  for more_example in more_examples:
    inputs = [(type_nparray(more_example[0][0][0]),), more_example[0][1]]
    outputs = (*map(type_nparray, more_example[1]),)
    cases.append([inputs, outputs])

  return cases

def decode_uids_test(cases):

  def _equal(a, b):
    # a, b: tuples of same length
    all_equal = True
    assert len(a) == len(b)
    for (aa, bb) in zip(a, b):
      # TODO(panos): consider checking with isinstance instead of type
      # type(np.int32((1,))) != type(np.int32((1)))
      assert type(aa) is type(bb), f"aa: {aa} {type(aa)}, bb: {bb} {type(bb)}"
      if isinstance(aa, (tf.Tensor, int, np.int32)):
        all_equal &= aa == bb
      elif isinstance(aa, np.ndarray):
        all_equal &= bool(np.equal(aa, bb))
      elif isinstance(aa, torch.Tensor):
        all_equal &= bool(aa == bb)
      else:
        raise NotImplementedError(f"{type(aa), type(bb)}")
    return all_equal

  for case in cases:
    # print('case', case)
    results = decode_uids(*case[0][0], **case[0][1])
    # print(results, case[1])
    if not _equal(results, case[1]):
      raise ValueError(f'Case failed.\ncase:\ninputs: {case[0]}\nexpected results: {case[1]}\nreal results: {results}')
  print(f"decode_uids: {len(cases)} test cases completed successfully.")


def read_and_decode_files():
  cpp_paths = glob.glob('/home/panos/git/p.meletis/panoptic_parts_datasets/tests/tests_files/gtFinePanopticParts/*/*/*.tif')
  cpp_spec = '/home/panos/git/github/pmeletis/metrics_design/panoptic_parts/panoptic_parts/specs/dataset_specs/cpp_datasetspec.yaml'
  ppp_paths = glob.glob('/home/panos/git/p.meletis/panoptic_parts_datasets/tests/tests_files/pascal_panoptic_parts/labels_v2/*/*.tif')
  ppp_spec = '/home/panos/git/github/pmeletis/metrics_design/panoptic_parts/panoptic_parts/specs/dataset_specs/ppp_datasetspec.yaml'

  cpp_spec = DatasetSpec(cpp_spec)
  ppp_spec = DatasetSpec(ppp_spec)

  for fp in cpp_paths:
    uids = np.asanyarray(Image.open(fp), dtype=np.int32)
    decode_uids(uids, return_sids_iids=True, return_sids_pids=True,
                experimental_dataset_spec=cpp_spec, experimental_correct_range=True)

  for fp in ppp_paths:
    uids = np.asanyarray(Image.open(fp), dtype=np.int32)
    decode_uids(uids, return_sids_iids=True, return_sids_pids=True,
                experimental_dataset_spec=ppp_spec, experimental_correct_range=True)


if __name__ == "__main__":
  decode_uids_test(decode_uids_cases())
  encode_ids_test(encode_ids_cases())
  read_and_decode_files()
