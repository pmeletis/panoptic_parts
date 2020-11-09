"""
This script computes the confusion matrix and prints segmentation metrics (mAcc, mIoU) for
ground truth and prediction files. The main components in this script are the following:
  - ground truth and prediction files for a dataset
  - an evaluation definition YAML file
  - the `ConfusionMatrixEvaluator` class from panoptic_parts.utils.experimental_evaluation

The `ConfusionMatrixEvaluator` class requires:
  1. a YAML evaluation definition filepath
  2. a list of ground truth and prediction paths pairs
  2. a reader function to read prediction paths and optionally convert them to a compatible format
     (see `ConfusionMatrixEvaluator` class documentation for more info)

This script generates these requirements as follows:
  1. The list of paths is created by `filepaths_pairs_fn`, change the function implementation
     according to your needs, note that `ConfusionMatrixEvaluator` requires only a list of path
     pairs, how it is created is up to the user.
  2. The reader function reads the predictions from the filepaths into an np.int32, np.ndarray
     with values in the evaluation ids range in `FILEPATH_EVALUATION_DEF`. We provide an example
     implementation `reader_fn`. Implement your reader function according to your
     predictions values. Definitions from `FILEPATH_EVALUATION_DEF` can be used.

Examples:
  1. Compute part-level class IoUs and mIoU for Cityscapes Panoptic Parts 24 parts classes:
     a. change `FILEPATH_PATTERN_GT_CPP` and `BASEPATH_PRED` and adapt the `filepaths_pairs_fn`
        function code as needed
     b. use FILEPATH_EVALUATION_DEF = 'panoptic_parts/evaluation/defs/cpp_parts_24.yaml' and
        adapt the `pred_reader_fn` function code as needed
     c. Run from the top-level directory panoptic_parts this script as:
        python -m panoptic_parts.evaluation.experimental_eval_segmentation_parts


This is the Tensorflow CPU implementation for faster, concurrent implementation.
Benchmarks (full resolution):
  1. ~35 sec for Cityscapes Panoptic Parts val set evaluation.
  2. ~30 sec for PASCAL Panoptic Parts val set evaluation.
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
from panoptic_parts.utils.experimental_evaluation import (
    ConfusionMatrixEvaluator, parse_sid_pid2eval_id)


FILEPATH_EVALUATION_DEF = 'panoptic_parts/evaluation/defs/cpp_parts_24.yaml'
FILEPATH_PATTERN_GT_CPP = op.join('tests', 'tests_files', 'cityscapes_panoptic_parts', 'gtFine', 'val', '*', '*.tif')
# FILEPATH_PATTERN_GT_PPP = op.join('tests', 'tests_files', 'pascal_panoptic_parts', 'labels', '*.tif')
BASEPATH_PRED = '/use/a/real/path'

def filepaths_pairs_fn(filepath_pattern_gt, basepath_pred):
  # return a list of tuples with paths
  filepaths_gt = glob.glob(filepath_pattern_gt)
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

filepaths_pairs = filepaths_pairs_fn(FILEPATH_PATTERN_GT_CPP, BASEPATH_PRED)

# preparation for evaluation
with open(FILEPATH_EVALUATION_DEF) as fd:
  defs = yaml.load(fd)
sid_pid2eval_id = defs['sid_pid2eval_id']
sid_pid2eval_id__values = sid_pid2eval_id.values()
Nclasses = len(set(sid_pid2eval_id__values))
ignore_ids = [max(sid_pid2eval_id__values) + 1] if -1 in sid_pid2eval_id__values else list()
names = list(defs['eval_id2name'].values()) if 'eval_id2name' in defs.keys() else None

#########
# here we assume that predictions are encoded as ground truth
def pred_reader_fn(fp_pred, sid_pid2eval_id):
  # function provided to ConfusionMatrixEvaluator,
  # see class documentation for more information
  label_pred = np.asarray(Image.open(fp_pred), dtype=np.int32)
  # here we assume that predictions have the same format as ground truth
  _, _, _, sids_pids_pred = decode_uids(label_pred, return_sids_pids=True)
  eids_pred = sid_pid2eval_id[sids_pids_pred]
  return eids_pred

max_sid = defs['max_sid']
sid_pid2eval_id = parse_sid_pid2eval_id(sid_pid2eval_id, max_sid)
sid_pid2eval_id__dense = _sparse_ids_mapping_to_dense_ids_mapping(
    sid_pid2eval_id, 0, length=(max_sid * 100 + 99) + 1)
pred_reader_fn = functools.partial(pred_reader_fn, sid_pid2eval_id=sid_pid2eval_id__dense)
#########

# create and run evaluator
evaluator = ConfusionMatrixEvaluator(FILEPATH_EVALUATION_DEF,
                                     filepaths_pairs,
                                     pred_reader_fn,
                                     experimental_validate_args=False)
cm = evaluator.compute_cm()
evaluator.print_metrics(names=names, printcmd=True, ignore_ids=ignore_ids)
