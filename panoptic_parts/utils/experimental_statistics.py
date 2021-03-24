import multiprocessing

import numpy as np
from PIL import Image
import tensorflow as tf

from panoptic_parts.utils.format import decode_uids


def reader_fn_default(filepath):
  uids = np.asarray(Image.open(filepath), dtype=np.int32)
  return uids


class DatasetStatisticsAggregator(object):

  def __init__(self,
               filepaths,
               reader_fn,
               init_states_reduce_fns):
    """
    Args:
      filepaths: an iterator of filepaths, each path will be passed to reader_fn
      reader_fn: a function with a path as input and an np.ndarray as output,
        with signature: path -> element_np
      init_states_reduce_fns: a list of tuples [(init_state, reduce_fn), ...],
        init_state: anything that tf.data.dataset.reduce accepts, e.g. a dict,
        reduce_fn: function that will be applied in parallel to the TF converted
          output of reader_fn, with signature: (old_state_tf, new_element_tf) -> new_state_tf
    """
    self.filepaths = filepaths
    self.reader_fn = reader_fn
    # self.init_states can definitely be a list but it seems that tf.data does not
    # work well with lists, e.g., if init_states = [one_init_state] it doesn't work
    self.init_states = {i: init_reduce[0] for i, init_reduce in enumerate(init_states_reduce_fns)}
    self.reduce_fns = [init_reduce[1] for init_reduce in init_states_reduce_fns]
    self.filepath_generator = lambda : (p for p in self.filepaths)
    self.num_parallel_calls = multiprocessing.cpu_count() - 1

  def _reduce_fns_aggregator(self, old_states, label_tf):
    new_states = {i: reduce_fn(old_states[i], label_tf)
                  for i, reduce_fn in enumerate(self.reduce_fns)}
    return new_states

  def _reader_fn_wrapper(self, filepath):

    def _wrapper(filepath):
      filepath = filepath.numpy().decode('utf-8')
      return self.reader_fn(filepath)

    labels_tf = tf.py_function(_wrapper, [filepath], tf.int32)
    return labels_tf

  def compute_statistics(self):
    """
    Returns:
      stats: a list of final states corresponding to the structure of given
        initial states [final_state, ...]
    """
    dataset = tf.data.Dataset.from_generator(self.filepath_generator,
                                             tf.string,
                                             output_shapes=())
    dataset = dataset.map(self._reader_fn_wrapper, num_parallel_calls=self.num_parallel_calls)
    dataset = dataset.reduce(self.init_states, self._reduce_fns_aggregator)
    # aggregated_states = list(map(lambda v: v.numpy(), dataset.values()))
    stats = list(dataset.values())
    return stats


def num_instances_per_image_state_and_reducer(Nimages):
  """
  Args:
    Nimages: the number of images in the dataset
  """
  # Nimages is the number of images in the dataset, it needs to be statically known
  #   since we will count instances for each image, and the state structure
  #   needs to be predefined
  init_state = {
      'instances_per_image': np.zeros(Nimages, dtype=np.int64),
      'label_counter': np.zeros((), dtype=np.int32)}

  def reduce_fn(old_state, label_tf):
    _, iids, _ = decode_uids(label_tf)
    # iids_unique exampes: [-1, 0, 3], [-1], [0, 2]
    iids_unique = tf.sort(tf.unique(tf.reshape(iids, [-1]))[0])
    # cannot take max(iids_unique) because iids may be not continuous integers
    num_iids = tf.shape(iids_unique)[0]
    predicate = tf.logical_and(tf.equal(iids_unique[0], -1), tf.shape(iids_unique) >= 2)
    num_iids = tf.cond(predicate, true_fn=lambda: num_iids-1, false_fn=lambda: num_iids)
    instances_per_image = old_state['instances_per_image']
    label_counter = old_state['label_counter']
    instances_per_image = tf.tensor_scatter_nd_add(
        instances_per_image, [[label_counter]], [num_iids])
    new_state = {'instances_per_image': instances_per_image,
                 'label_counter': label_counter + 1}
    return new_state
  
  return (init_state, reduce_fn)
