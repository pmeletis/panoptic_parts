import glob
import itertools
from datetime import datetime
import os.path as op

import numpy as np

from panoptic_parts.utils.format import decode_uids
from panoptic_parts.utils.experimental_statistics import (
    DatasetStatisticsAggregator,
    num_instances_per_image_state_and_reducer,
    reader_fn_default)


# CPP: num instances: 42674, num instances per image: 12.28
BASEPATH = 'tests/tests_files/gtFinePanopticParts'
patterns = [op.join(BASEPATH, 'train/*/*.tif'), op.join(BASEPATH, 'val/*/*.tif')]
label_filepaths = list(itertools.chain.from_iterable(glob.glob(fp) for fp in patterns))
assert len(label_filepaths) == 3475

def read_label(filepath):
  from PIL import Image
  now = datetime.now()
  uids = np.asarray(Image.open(filepath), dtype=np.int32)
  print(datetime.now() - now)
  return uids

init_state, reduce_fn = num_instances_per_image_state_and_reducer(len(label_filepaths))

statistics = DatasetStatisticsAggregator(
    label_filepaths, read_label, [(init_state, reduce_fn)])

stats = statistics.compute_statistics()
instances_per_image = stats[0]['instances_per_image']
print(
    f"Total number of instances: {np.sum(instances_per_image)}.",
    '\n',
    f"Average number of instances per image: {np.mean(instances_per_image)}.",
    sep='')
