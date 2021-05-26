"""
This script computes the confusion matrix and prints segmentation metrics (mAcc, mIoU) for
ground truth and prediction files. The main components in this script are the following:
  - ground truth and prediction files for a dataset
  - an evaluation definition YAML file
  - the `ConfusionMatrixEvaluator` class from panoptic_parts.utils.experimental_evaluation_IOU

The `ConfusionMatrixEvaluator` class requires:
  1. a YAML evaluation specification
  2. a list of ground truth and prediction paths pairs
  3. a reader function to read prediction paths and optionally convert them to a compatible format
     (see `ConfusionMatrixEvaluator` class documentation for more info)

This script generates the aboves elements as follows:
  2. The list of paths is created by `filepaths_pairs_fn`, change the function implementation
     according to your needs, note that `ConfusionMatrixEvaluator` requires only a list of path
     pairs, how it is created is up to the user.
  3. The reader function reads the predictions from the filepaths into an np.int32, np.ndarray
     with values in the evaluation ids range in `FILEPATH_EVALUATION_DEF`. We provide an example
     implementation `reader_fn`. Implement your reader function according to your
     predictions values. Definitions from `FILEPATH_EVALUATION_DEF` can be used.

Examples:
  1. Compute part-level class IoUs and mIoU for Cityscapes Panoptic Parts 24 parts classes:
     a. change `FILEPATH_PATTERN_GT_CPP` and `BASEPATH_PRED` and adapt the `filepaths_pairs_fn`
        function code as needed
     b. use FILEPATH_EVALUATION_DEF = 'cpp_iouparts_24_evalspec.yaml' and
        adapt the `pred_reader_fn` function code as needed
     c. Run from the top-level directory this script as:
        python -m eval_segmentation_parts


This is the Tensorflow 2.x CPU implementation for faster, concurrent implementation.
Benchmarks (full resolution, Intel® Core™ i9-7900X CPU @ 3.30GHz):
  1. ~35 sec for Cityscapes Panoptic Parts val set evaluation.
  2. ~30 sec for PASCAL Panoptic Parts val set evaluation.
"""
import glob
import os.path as op
import multiprocessing
import functools

import yaml
from PIL import Image
import numpy as np
import tensorflow as tf
assert tf.version.VERSION[0] == '2', 'Uses TF r2.x functionality.'

from panoptic_parts.utils.format import decode_uids
from panoptic_parts.utils.experimental_evaluation_IOU import ConfusionMatrixEvaluator_v2
from panoptic_parts.specs.eval_spec import SegmentationPartsEvalSpec


FILEPATH_EVALUATION_DEF = 'cpp_iouparts_24_evalspec.yaml'
FILEPATH_PATTERN_GT_CPP = op.join('/home/panos/git/p.meletis/panoptic_parts_datasets', 'tests', 'tests_files',
    'cityscapes_panoptic_parts', 'gtFine_v2', 'val', '*', '*.tif')
# FILEPATH_PATTERN_GT_PPP = op.join('/home/panos/git/p.meletis/panoptic_parts_datasets', 'tests', 'tests_files',
#     'pascal_panoptic_parts', 'labels', '*.tif')
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
    # assert False, 'delete this when adapted to your needs'
    ########################
    pairs.append((fp_gt, fp_pred))
  return pairs

######################
# Adapt to your case #
# here we assume that predictions are encoded as ground truth
def pred_reader_fn(fp_pred, sid_pid2eval_id):
  # function provided to ConfusionMatrixEvaluator,
  # see class documentation for more information
  label_pred = np.asarray(Image.open(fp_pred), dtype=np.int32)
  # here we assume that predictions have the same format as ground truth
  _, _, _, sids_pids_pred = decode_uids(label_pred, return_sids_pids=True)
  eids_pred = sid_pid2eval_id[sids_pids_pred]
  return eids_pred
######################

# create and run evaluator
spec = SegmentationPartsEvalSpec(FILEPATH_EVALUATION_DEF)
filepaths_pairs = filepaths_pairs_fn(FILEPATH_PATTERN_GT_CPP, BASEPATH_PRED)
pred_reader_fn = functools.partial(pred_reader_fn, sid_pid2eval_id=spec.sp2e_np)
evaluator = ConfusionMatrixEvaluator_v2(spec,
                                        filepaths_pairs,
                                        pred_reader_fn,
                                        experimental_validate_args=False)
cm = evaluator.compute_cm()
evaluator.print_metrics(names=spec.scene_part_classes,
                        printcmd=True,
                        ignore_ids=[spec.eid_ignore])
