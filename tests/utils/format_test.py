
import sys
assert float(sys.version[:3]) >= 3.7, 'This test uses Python >= 3.7 functionality.'
import functools

import numpy as np
import tensorflow as tf
assert tf.version.VERSION[0] == '2', 'This test uses TF r2.x functionality.'

from utils.format import decode_uids, encode_ids

# TOTAL_TESTS = 96 + 24 = 120

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
              np.int32]

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
  print(f"{len(cases)} test cases completed successfully.")

def decode_uids_cases():
  ersiT = {'experimental_return_sids_iids': True}
  erspT = {'experimental_return_sids_pids': True}
  ersipT = {'experimental_return_sids_iids': True, 'experimental_return_sids_pids': True}
  # examples = [example, ...]
  # example = [inputs, outputs] = [[args, kwargs], outputs]
  examples = [
      [[(1,), {}], (1, -1, -1)],
      [[(1,), ersiT], (1, -1, -1, 1)],
      [[(1,), erspT], (1, -1, -1, 100)],
      [[(1,), ersipT], (1, -1, -1, 1, 100)],
      [[(11,), {}], (11, -1, -1)],
      [[(11,), ersiT], (11, -1, -1, 11)],
      [[(11,), erspT], (11, -1, -1, 1100)],
      [[(11,), ersipT], (11, -1, -1, 11, 1100)],
      [[(1_002,), {}], (1, 2, -1)],
      [[(1_002,), ersiT], (1, 2, -1, 1002)],
      [[(1_002,), erspT], (1, 2, -1, 100)],
      [[(1_002,), ersipT], (1, 2, -1, 1002, 100)],
      [[(11_002,), {}], (11, 2, -1)],
      [[(11_002,), ersiT], (11, 2, -1, 11002)],
      [[(11_002,), erspT], (11, 2, -1, 1100)],
      [[(11_002,), ersipT], (11, 2, -1, 11002, 1100)],
      [[(1_002_03,), {}], (1, 2, 3)],
      [[(1_002_03,), ersiT], (1, 2, 3, 1002)],
      [[(1_002_03,), erspT], (1, 2, 3, 103)],
      [[(1_002_03,), ersipT], (1, 2, 3, 1002, 103)],
      [[(11_002_03,), {}], (11, 2, 3)],
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
      np.int32
  ]

  # cases: [[inputs, outputs], ...] = [[[args, kwargs], outputs], ...]
  cases = list()
  for example in examples:
    for type_fn in types_fn:
      inputs = [(type_fn(example[0][0][0]),), example[0][1]]
      outputs = (*map(type_fn, example[1]),)
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
        all_equal &= np.all(np.equal(aa, bb))
      else:
        raise NotImplementedError(f"{type(aa), type(bb)}")
    return all_equal

  for case in cases:
    # print('case', case)
    results = decode_uids(*case[0][0], **case[0][1])
    # print(results, case[1])
    if not _equal(results, case[1]):
      print(case, results)
  print(f"{len(cases)} test cases completed successfully.")

if __name__ == "__main__":
  decode_uids_test(decode_uids_cases())
  encode_ids_test(encode_ids_cases())
