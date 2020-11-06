"""
This script saves visualized PASCAL Panoptic Parts labels on three levels:
  semantic, semantic-instance, and semantic-instance-parts
It globs the FILEPATH_PATTERN_LABELS pattern to find labels and uses the
`panoptic_parts/utils/defs/ppp_100_72.yaml` task definition for visualization.
Visualization will be written in a directory named `labels_colored` aside
the `labels` directory (here `tests/tests_files/pascal_panoptic_parts/labels_colored`).

Benchmark:
  1. ~50 sec to write 5105 visualized labels (val set)
"""

import glob
from multiprocessing import Pool

import yaml
import numpy as np
from PIL import Image

from panoptic_parts.utils.utils import _transform_uids, safe_write
from panoptic_parts.utils.visualization import experimental_colorize_label

FILEPATH_PATTERN_LABELS = 'tests/tests_files/pascal_panoptic_parts/labels/*.tif'

filepaths_labels = glob.glob(FILEPATH_PATTERN_LABELS)
# filepaths_images = [fl.replace('/labels/', '/images/').replace('.tif', '.jpg')
#                     for fl in filepaths_labels]
filepaths_images = [None] * len(filepaths_labels) # unused for now
basepaths_outputs = [fl.replace('/labels/', '/labels_colored/').replace('.tif', '')
                     for fl in filepaths_labels]

FILEPATH_TASK_DEF = 'panoptic_parts/utils/defs/ppp_100_72.yaml'

with open(FILEPATH_TASK_DEF) as fp:
  task_def = yaml.load(fp)
SID2COLOR = task_def['sid2color']
# add colors for all sids that may exist in labels, but don't have a color from task_def
SID2COLOR.update({sid: SID2COLOR[-1] # we use the void color here
                  for sid in range(task_def['max_sid'])
                  if not (sid in task_def['valid_sids'] or sid in SID2COLOR)})
MAX_SID = task_def['max_sid']
SID2PIDS_GROUPS = task_def['sid2pids_groups']


def write_fn(args):
  _, label_path, basepath_output = args

  uids = np.asarray(Image.open(label_path), dtype=np.int32)
  uids = _transform_uids(uids, MAX_SID, SID2PIDS_GROUPS)
  uids_sem_inst_parts_colored, uids_sem_colored, uids_sem_inst_colored  = \
      experimental_colorize_label(uids,
                                  sid2color=SID2COLOR,
                                  return_sem=True,
                                  return_sem_inst=True,
                                  emphasize_instance_boundaries=True)

  safe_write(basepath_output + '_sem_inst_parts_colored.png', uids_sem_inst_parts_colored)
  safe_write(basepath_output + '_sem_colored.png', uids_sem_colored)
  safe_write(basepath_output + '_sem_inst_colored.png', uids_sem_inst_colored)

  return True


args_zip = zip(filepaths_images, filepaths_labels, basepaths_outputs)
with Pool() as pool:
  results = tuple(pool.imap_unordered(write_fn, args_zip, chunksize=10))

print(f"{len(tuple(filter(None, results)))} labels out of {len(results)} were successfully saved.")
