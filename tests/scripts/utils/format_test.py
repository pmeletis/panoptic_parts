
import functools

import numpy as np
import tensorflow as tf
assert tf.version.VERSION[0] == '2', 'This test uses TF r2.x functionality.'

from scripts.utils.format import encode_ids

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

def 

if __name__ == "__main__":
  encode_ids_test(encode_ids_cases())
