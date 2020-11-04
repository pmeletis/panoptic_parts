import multiprocessing

import numpy as np
import tensorflow as tf
assert tf.version.VERSION[0] == '2', 'Uses TF r2.x functionality.'

from panoptic_parts.utils.utils import _print_metrics_from_confusion_matrix

# VALIDATE_ARGS = True


class ConfusionMatrixEvaluator(object):
  """
  Computes the confusion matrix for the provided ground truth and prediction pairs filepaths
  using a tf.data pipeline for fast execution.

  A standard use of this class is:
    evaluator = ConfusionMatrixEvaluator(list_of_filepaths_pairs, reader_fn, Nclasses)
    confusion_matrix = evaluator.compute_cm()
    metrics = compute_metrics_with_any_external_function(confusion_matrix)
  """

  def __init__(self, filepaths_pairs, reader_fn, Nclasses, experimental_validate_args=False):
    """
    Args:
      filepaths_pairs: [(path_gt, path_pred), ...], each pair will be passed to reader_fn
      reader_fn: a function with inputs and outputs tf.Tensor s and signature
        (path_gt_tensor, path_pred_tensor) -> (gt_tensor, pred_tensor),
        gt_tensor and pred_tensor must be in range [0, Nclasses-1], i.e., the eval_ids
      Nclasses: the number of distinct integer ids in gt_tensor and pred_tensor
      experimental_validate_args: will perform validation of tensor values to catch errors and show
        meaningful explanations, disabled by default since it is computationally intensive
    """
    self.filepaths_pairs = filepaths_pairs
    self.filepaths_pairs_generator = lambda : (p for p in self.filepaths_pairs)
    self.reader_fn = reader_fn
    self.Nclasses = Nclasses
    self.experimental_validate_args = experimental_validate_args
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

  def compute_cm(self):
    dataset = tf.data.Dataset.from_generator(self.filepaths_pairs_generator,
                                             (tf.string, tf.string),
                                             output_shapes=((), ()))
    dataset = dataset.map(self.reader_fn, num_parallel_calls=self.num_parallel_calls)
    dataset = dataset.reduce(self.confusion_matrix, self._reduce_func)
    self.confusion_matrix = dataset.numpy()
    return self.confusion_matrix

  def print_metrics(self, *args, **kwargs):
    # if self.confusion_matrix is None:
    #   raise AttributeError('Run .evaluate() method first.')
    _print_metrics_from_confusion_matrix(self.confusion_matrix, *args, **kwargs)

  # TODO(panos): move functionality from _print_metrics_from_confusion_matrix to this class
