"""
Library for IOU-based evaluation functions.
"""
import multiprocessing

from PIL import Image
import numpy as np
import tensorflow as tf
assert tf.version.VERSION[0] == '2', 'Uses TF r2.x functionality.'
import yaml
from typing import Dict, Union

from panoptic_parts.utils.format import decode_uids
from panoptic_parts.utils.utils import (
    _print_metrics_from_confusion_matrix,
    _sparse_ids_mapping_to_dense_ids_mapping,
    parse__sid_pid2eid__v2)

# VALIDATE_ARGS = True


def parse_sid_pid2eval_id(sid_pid2eval_id: Dict[int, int], max_sid: int):
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


def _parse_yaml(fp_yaml):
  with open(fp_yaml) as fd:
    defs = yaml.load(fd, Loader=yaml.Loader)
  max_sid = defs['max_sid']
  sid_pid2eval_id = defs['sid_pid2eval_id']
  sid_pid2eval_id = parse_sid_pid2eval_id(sid_pid2eval_id, max_sid)
  sid_pid2eval_id__dense = _sparse_ids_mapping_to_dense_ids_mapping(
      sid_pid2eval_id, 0, length=(max_sid * 100 + 99) + 1)
  # Nclasses = len(set(sid_pid2eval_id.values()))
  return sid_pid2eval_id__dense


class ConfusionMatrixEvaluator(object):
  """
  Computes the confusion matrix for the provided ground truth and prediction pairs filepaths
  using a tf.data pipeline for fast execution.

  A standard use of this class is:
    evaluator = ConfusionMatrixEvaluator(filepath_yaml, list_of_filepaths_pairs, pred_reader_fn)
    confusion_matrix = evaluator.compute_cm()
    metrics = compute_metrics_with_any_external_function(confusion_matrix)
  """

  def __init__(self,
               filepath_yaml,
               filepaths_pairs,
               pred_reader_fn,
               experimental_validate_args=False):
    """
    Args:
      filepath_yaml: a YAML definition from evaluation/defs
      filepaths_pairs: [(path_gt, path_pred), ...], each path_pred will be passed to pred_reader_fn
      pred_reader_fn: a function with a path string as input and a np.ndarray as output and
        signature: (path_pred) -> (pred_np), pred_np must be in range [0, Nclasses-1],
        i.e., the evaluation ids in YAML definition
      experimental_validate_args: will perform validation of tensor values to catch errors and show
        meaningful explanations, disabled by default since it is computationally intensive
    """
    self.filepath_yaml = filepath_yaml
    self.filepaths_pairs = filepaths_pairs
    self.pred_reader_fn = pred_reader_fn
    self.experimental_validate_args = experimental_validate_args

    self.filepaths_pairs_generator = lambda : (p for p in self.filepaths_pairs)
    self.sid_pid2eval_id__dense = _parse_yaml(self.filepath_yaml)
    self.Nclasses = len(np.unique(self.sid_pid2eval_id__dense))
    self.confusion_matrix = np.zeros((self.Nclasses, self.Nclasses), dtype=np.int64)
    self.num_parallel_calls = multiprocessing.cpu_count() - 1

  def _assertions_for_value_range(self, gt_flat, pred_flat):
    ids_gt_unique = tf.unique(gt_flat)[0]
    ids_pred_unique = tf.unique(pred_flat)[0]
    valid_ids = tf.range(self.Nclasses)
    ms = lambda c: (
        f"The reader function must convert {'ground truth' if c=='gt' else 'predictions'}"
        f" to continuous ids in range [0, {self.Nclasses-1}]. Found ids outside this"
        f" range. The following condition asserts if set("
        f"{'ground truth' if c=='gt' else 'predictions'} ids) == set(range(Nclasses)).")
    is_subset = lambda b, a: tf.equal(tf.shape(tf.sets.difference(b, a))[1], 0)
    assertions = [
        # extra [] because tf.sets.difference doesn't accept rank 1 tensors (that's a TF bug)
        tf.debugging.assert_equal(is_subset([ids_gt_unique], [valid_ids]), True, message=ms('gt')),
        tf.debugging.assert_equal(is_subset([ids_pred_unique], [valid_ids]), True, message=ms('pr'))]
    return assertions

  def _reduce_func(self, old_state, gt_pred):
    _flat = lambda x: tf.reshape(x, [-1])
    gt_flat, pred_flat = map(_flat, gt_pred)
    if self.experimental_validate_args:
      with tf.control_dependencies(self._assertions_for_value_range(gt_flat, pred_flat)):
        gt_flat = tf.identity(gt_flat)
    cm = tf.math.confusion_matrix(gt_flat, pred_flat,
                                  num_classes=self.Nclasses, dtype=tf.int64)
    new_state = old_state + cm
    return new_state

  def _readers_fn(self, fp_gt, fp_pred):

    def _read_gt_py(fp_gt):
      # eager function, required since .tif images can only be opened by Pillow for now
      fp_gt = fp_gt.numpy().decode('utf-8')
      label_gt = np.asarray(Image.open(fp_gt), dtype=np.int32)
      _, _, _, sids_pids_gt = decode_uids(label_gt, return_sids_pids=True)
      eids_gt = self.sid_pid2eval_id__dense[sids_pids_gt]
      return eids_gt

    eids_gt = tf.py_function(_read_gt_py, [fp_gt], tf.int32)

    def _wrapper(fp_pred):
      fp_pred = fp_pred.numpy().decode('utf-8')
      return self.pred_reader_fn(fp_pred)

    eids_pred = tf.py_function(_wrapper, [fp_pred], tf.int32)

    return eids_gt, eids_pred

  def compute_cm(self):
    dataset = tf.data.Dataset.from_generator(self.filepaths_pairs_generator,
                                             (tf.string, tf.string),
                                             output_shapes=((), ()))
    dataset = dataset.map(self._readers_fn, num_parallel_calls=self.num_parallel_calls)
    dataset = dataset.reduce(self.confusion_matrix, self._reduce_func)
    self.confusion_matrix = dataset.numpy()
    return self.confusion_matrix

  def print_metrics(self, *args, **kwargs):
    # if self.confusion_matrix is None:
    #   raise AttributeError('Run .evaluate() method first.')
    _print_metrics_from_confusion_matrix(self.confusion_matrix, *args, **kwargs)

  # TODO(panos): move functionality from _print_metrics_from_confusion_matrix to this class


class ConfusionMatrixEvaluator_v2(object):
  """
  Computes the confusion matrix for the provided ground truth and prediction pairs filepaths
  using a tf.data pipeline for fast execution.

  A standard use of this class is:
    evaluator = ConfusionMatrixEvaluator_v2(eval_spec, list_of_filepaths_pairs, pred_reader_fn)
    confusion_matrix = evaluator.compute_cm()
    metrics = compute_metrics_with_any_external_function(confusion_matrix)
  """

  def __init__(self,
               eval_spec,
               filepaths_pairs,
               pred_reader_fn,
               experimental_validate_args=False):
    """
    Args:
      eval_spec: an EvalSpec object, containing attributes: Nclasses, sp2e_np
      filepaths_pairs: [(path_gt, path_pred), ...], each path_pred will be passed to pred_reader_fn
      pred_reader_fn: a function with a path string as input and a np.ndarray as output and
        signature: pred_reader_fn(path_pred) -> pred_np,
        pred_np must have values in range [0, Nclasses-1], i.e. the eval ids in YAML definition
      experimental_validate_args: will perform validation of tensor values to catch errors and show
        meaningful explanations, disabled by default since it is computationally intensive
    """
    self.spec = eval_spec
    self.filepaths_pairs = filepaths_pairs
    self.pred_reader_fn = pred_reader_fn
    self.experimental_validate_args = experimental_validate_args

    self.filepaths_pairs_generator = lambda : (p for p in self.filepaths_pairs)
    self.Nclasses = eval_spec.Nclasses
    self.confusion_matrix = np.zeros((self.Nclasses, self.Nclasses), dtype=np.int64)
    self.num_parallel_calls = multiprocessing.cpu_count() - 1

  def _assertions_for_value_range(self, gt_flat, pred_flat):
    ids_gt_unique = tf.unique(gt_flat)[0]
    ids_pred_unique = tf.unique(pred_flat)[0]
    valid_ids = tf.range(self.Nclasses)
    ms = lambda c: (
        f"The reader function must convert {'ground truth' if c=='gt' else 'predictions'}"
        f" to continuous ids in range [0, {self.Nclasses-1}]. Found ids outside this"
        f" range. The following condition asserts if set("
        f"{'ground truth' if c=='gt' else 'predictions'} ids) == set(range(Nclasses)).")
    is_subset = lambda b, a: tf.equal(tf.shape(tf.sets.difference(b, a))[1], 0)
    assertions = [
        # extra [] because tf.sets.difference doesn't accept rank 1 tensors (that's a TF bug)
        tf.debugging.assert_equal(is_subset([ids_gt_unique], [valid_ids]), True, message=ms('gt')),
        tf.debugging.assert_equal(is_subset([ids_pred_unique], [valid_ids]), True, message=ms('pr'))]
    return assertions

  def _reduce_func(self, old_state, gt_pred):
    _flat = lambda x: tf.reshape(x, [-1])
    gt_flat, pred_flat = map(_flat, gt_pred)
    if self.experimental_validate_args:
      with tf.control_dependencies(self._assertions_for_value_range(gt_flat, pred_flat)):
        gt_flat = tf.identity(gt_flat)
    cm = tf.math.confusion_matrix(gt_flat, pred_flat,
                                  num_classes=self.Nclasses, dtype=tf.int64)
    new_state = old_state + cm
    return new_state

  def _readers_fn(self, fp_gt, fp_pred):

    def _read_gt_py(fp_gt):
      # eager function, required since .tif images can only be opened by Pillow for now
      fp_gt = fp_gt.numpy().decode('utf-8')
      label_gt = np.asarray(Image.open(fp_gt), dtype=np.int32)
      _, _, _, sids_pids_gt = decode_uids(label_gt, return_sids_pids=True)
      eids_gt = self.spec.sp2e_np[sids_pids_gt]
      return eids_gt

    def _wrapper(fp_pred):
      fp_pred = fp_pred.numpy().decode('utf-8')
      return self.pred_reader_fn(fp_pred)

    eids_gt = tf.py_function(_read_gt_py, [fp_gt], tf.int32)
    eids_pred = tf.py_function(_wrapper, [fp_pred], tf.int32)

    return eids_gt, eids_pred

  def compute_cm(self):
    dataset = tf.data.Dataset.from_generator(self.filepaths_pairs_generator,
                                             (tf.string, tf.string),
                                             output_shapes=((), ()))
    dataset = dataset.map(self._readers_fn,
                          num_parallel_calls=self.num_parallel_calls,
                          deterministic=False)
    dataset = dataset.reduce(self.confusion_matrix, self._reduce_func)
    self.confusion_matrix = dataset.numpy()
    return self.confusion_matrix

  def print_metrics(self, *args, **kwargs):
    # if self.confusion_matrix is None:
    #   raise AttributeError('Run .evaluate() method first.')
    _print_metrics_from_confusion_matrix(self.confusion_matrix, *args, **kwargs)

  # TODO(panos): move functionality from _print_metrics_from_confusion_matrix to this class


if __name__ == '__main__':
  with open('../cpp_parts_24.yaml') as fd:
    spec = yaml.load(fd)
  sp2e_new = parse__sid_pid2eid__v2(spec['sid_pid2eid__template'])
  eval_id_max_non_ignored = max(sp2e_new.values())
  # prints (sid_pid, eval_id) without the tuples containing the background (0) and the ignored eids
  print(*filter(lambda e: 0 < e[1] < eval_id_max_non_ignored, sp2e_new.items()), sep='\n')
  breakpoint()
