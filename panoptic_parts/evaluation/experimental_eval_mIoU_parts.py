"""
This script computes the confusion matrix and prints related metrics (e.g. mIoU) for pairs of
ground truth and prediction files. The main components to use this script are the following:
  - ground truth and prediction files for a dataset
  - an evaluation definition YAML file
  - the `ConfusionMatrixEvaluator` class from panoptic_parts.utils.experimental_evaluation

The `ConfusionMatrixEvaluator` class requires:
  1. a list of ground truth and prediction paths pairs
  2. a reader function to read these paths and optionally convert them to a compatible format
     (see `ConfusionMatrixEvaluator` class documentation for more info)
  3. the number of classes for initializing the confusion matrix

This script generates these requirements as follows:
  1. The list of paths is created by `filepaths_pairs_fn` using `BASEPATH_GT` and `BASEPATH_PRED`,
     change these paths and the function implementation according to your needs, note that
     `ConfusionMatrixEvaluator` requires only a list of path pairs, how it is created is up
     to the user.
  2. The reader function reads the labels from the filepaths into tf.int32 tf.Tensor s with
     values from the evaluation ids in `FILEPATH_EVALUATION_DEF`. We provide an example
     implementation `reader_fn`. Implement your reader function according to your
     predictions values. Definitions from `FILEPATH_EVALUATION_DEF` can be used.
  3. The number of classes are computed using only `FILEPATH_EVALUATION_DEF`.

Examples:
  1. Compute per-class and mIoU for Cityscapes Panoptic Parts 24 parts classes:
     a. change `BASEPATH_GT` and `BASEPATH_PRED` and adapt the `filepaths_pairs_fn`
        function code section denoted by ########
     b. use FILEPATH_EVALUATION_DEF = 'panoptic_parts/evaluation/defs/cpp_parts_24.yaml' and
        adapt the `read_filepaths_and_convert` function code section denoted by ########,
        e.g. if predictions are already encoded using the evaluation ids from
        FILEPATH_EVALUATION_DEF, then just delete the lines in adapt section
     c. Run from the top-level directory panoptic_parts the script as:
        python -m panoptic_parts.evaluation.experimental_eval_mIoU_parts


This is the Tensorflow CPU implementation for faster, concurrent implementation.
Benchmarks: ~35 sec for full resolution Cityscapes evaluation.
"""

import glob
import os.path as op
import multiprocessing
import functools

from PIL import Image
import numpy as np
import tensorflow as tf
assert tf.version.VERSION[0] == '2', 'Uses TF r2.x functionality.'

import yaml
from panoptic_parts.utils.utils import _sparse_ids_mapping_to_dense_ids_mapping
from panoptic_parts.utils.format import decode_uids
from panoptic_parts.utils.experimental_evaluation import ConfusionMatrixEvaluator


FILEPATH_EVALUATION_DEF = 'panoptic_parts/evaluation/defs/cpp_parts_24.yaml'
BASEPATH_GT = op.join('tests', 'tests_files', 'cityscapes_panoptic_parts', 'gtFine', 'val')
# use a real path with predictions
BASEPATH_PRED = BASEPATH_GT


def filepaths_pairs_fn(basepath_gt, basepath_pred):
  # return a list of tuples with paths
  filepaths_gt = glob.glob(op.join(basepath_gt, '*', '*.tif'))
  print(f"Found {len(filepaths_gt)} ground truth labels.")
  pairs = list()
  for fp_gt in filepaths_gt:
    image_id = op.basename(fp_gt)[:-4]
    ########################
    # Adapt to your system #
    # here we use the ground truth paths for predictions
    fp_pred = op.join(basepath_pred, f"{image_id}.tif")
    fp_pred = fp_gt
    assert False, 'delete this when adapted to your needs'
    ########################
    pairs.append((fp_gt, fp_pred))
  return pairs

filepaths_pairs = filepaths_pairs_fn(BASEPATH_GT, BASEPATH_PRED)


def parse_sid_pid2eval_id(sid_pid2eval_id, max_sid):
  # this function fills in sid_pid2eval_id dictionary all missing sids_pids keys and
  # replaces -1 values with a new eval_id, see 1., 2., 3., 4. comments for steps
  # the returned dict contains all possible sids_pids keys, i.e., [0, (max_sid + 1) * 100]

  # 1. if key -1 exists, add all non-present sids, up to max_sid, with value of sid_pid2eval_id[-1]
  keys = sid_pid2eval_id.keys()
  if -1 in keys:
    for sid in range(max_sid + 1):
      if sid not in keys:
        sid_pid2eval_id[sid] = sid_pid2eval_id[-1]

  # 2. replace sids with all sid_pid s and same values
  # list() otherwise: RuntimeError: dictionary changed size during iteration
  keys = list(sid_pid2eval_id.keys())
  for key in keys:
    if 0 <= key <= 99:
      for pid in range(100):
        sid_pid = key * 100 + pid
        if sid_pid not in keys:
          sid_pid2eval_id[sid_pid] = sid_pid2eval_id[key]

  # 3. if key -1 exists, fill all not existing sid_pid s with value of sid_pid2eval_id[-1]
  keys = sid_pid2eval_id.keys()
  if -1 in keys:
    for sid in range(1, max_sid + 1):
      for pid in range(100):
        sid_pid = key * 100 + pid
        if sid_pid not in keys:
          sid_pid2eval_id[sid_pid] = sid_pid2eval_id[-1]
    del sid_pid2eval_id[-1]

  # 4. replace all -1 values with id_ignore
  id_ignore = max(sid_pid2eval_id.values()) + 1
  sid_pid2eval_id = {k: id_ignore if v==-1 else v for k, v in sid_pid2eval_id.items()}

  # TODO(panos): include assertion in _sparse_ids_mapping_to_dense_ids_mapping and make void optional
  assert set(sid_pid2eval_id.keys()) == set(range((max_sid + 1) * 100))

  return sid_pid2eval_id


def read_filepaths_and_convert(fp_gt, fp_pred, sid_pid2eval_id):
  # function provided to ConfusionMatrixEvaluator,
  # see class documentation for more information
  
  def _py_func(fp_gt, fp_pred):
    # eager function
    # needed because .tif images can only be opened by Pillow for now
    fp_gt = fp_gt.numpy().decode('utf-8')
    label_gt = np.asarray(Image.open(fp_gt), dtype=np.int32)
    _, _, _, sids_pids_gt = decode_uids(label_gt, return_sids_pids=True)
    sids_pids_gt = sid_pid2eval_id[sids_pids_gt]
    fp_pred = fp_pred.numpy().decode('utf-8')
    label_pred = np.asarray(Image.open(fp_pred), dtype=np.int32)
    ########################
    # Adapt to your system #
    # here we assume that predictions have the same format as ground truth
    _, _, _, sids_pids_pred = decode_uids(label_pred, return_sids_pids=True)
    sids_pids_pred = sid_pid2eval_id[sids_pids_pred]
    assert False, 'delete this when adapted to your needs'
    ########################
    return sids_pids_gt, sids_pids_pred

  return tf.py_function(_py_func, [fp_gt, fp_pred], (tf.int32, tf.int32))


# preparation for evaluation
with open(FILEPATH_EVALUATION_DEF) as fd:
  defs = yaml.load(fd)
max_sid = defs['max_sid']
sid_pid2eval_id = defs['sid_pid2eval_id']
sid_pid2eval_id = parse_sid_pid2eval_id(sid_pid2eval_id, max_sid)
sid_pid2eval_id__dense = _sparse_ids_mapping_to_dense_ids_mapping(
    sid_pid2eval_id, 0, length=(max_sid * 100 + 99) + 1)
reader_fn = functools.partial(read_filepaths_and_convert, sid_pid2eval_id=sid_pid2eval_id__dense)
Nclasses = len(set(sid_pid2eval_id.values()))
ignore_ids = [max(sid_pid2eval_id.values())] if -1 in defs['sid_pid2eval_id'].values() else list()
names = list(defs['eval_id2name'].values()) if 'eval_id2name' in defs.keys() else None

# create and run evaluator
evaluator = ConfusionMatrixEvaluator(filepaths_pairs, reader_fn, Nclasses)
cm = evaluator.compute_cm()
evaluator.print_metrics(names=names, printcmd=True, ignore_ids=ignore_ids)
