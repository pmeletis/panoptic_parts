"""
Run this script as
`python -m panoptic_parts.pascal_panoptic_parts.visualize_label_with_legend \
     <label_path> <datasetspec_path>`
to visualize a PASCAL-Panoptic-Parts label in all three levels (semantic, instance, parts),
together with a legend including all the colors and uids in that label.
"""

import os.path as op
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ruamel.yaml import YAML

from panoptic_parts.utils.visualization import experimental_colorize_label
from panoptic_parts.utils.format import decode_uids, encode_ids
from panoptic_parts.specs.dataset_spec import DatasetSpec


def visualize_from_paths(label_path, datasetspec_path):
  """
  Visualizes in a pyplot window a label from the provided path.
  For reading files, Pillow is used, so the path and format
  must be Pillow-compatible.

  For visualization pixels are colored:
    - semantic-level ids: according to colors defined in task_def
    - semantic-instance-level ids: with random shades of colors defined in task_def
    - semantic-instance-parts-level ids: with a mixture of parula colormap and the shades above
  See panoptic_parts.utils.visualization.uid2color for more information on color generation.

  Args:
    label_path: a label path, will be passed to Pillow.Image.open
    datasetspec_path: a YAML file path, including keys: `sid2color`, `max_sid`, `valid_sids`
  """
  spec = DatasetSpec(datasetspec_path)
  uids = np.array(Image.open(label_path), dtype=np.int32)
  # for PPP, we need to fold groupable parts
  uids = encode_ids(*decode_uids(uids, experimental_dataset_spec=spec))

  uids_sem_inst_parts_colored, uid2color_dct = experimental_colorize_label(
      uids, sid2color=spec.sid2scene_color, emphasize_instance_boundaries=True, return_uid2color=True)

  # plot
  _, ax1 = plt.subplots()

  # generate legend, h is a hidden rectangle just to create a legend entry
  handles = []
  handles_text = []
  uids_unique = np.unique(uids)
  for uid in uids_unique:
    h = plt.Rectangle((0, 0), 1, 1, fc=list(map(lambda x: x/255, uid2color_dct[uid])))
    handles.append(h)
    _, _, _, sid_pid = decode_uids(uid, return_sids_pids=True)
    scene_class_part_class = spec.scene_class_part_class_from_sid_pid(sid_pid)
    handles_text.append(f'{uid}: {scene_class_part_class}')

  ax1.imshow(uids_sem_inst_parts_colored)
  ax1.set_title('labels colored on semantic, instance, and part levels', fontsize='small')
  ax1.legend(handles, handles_text, ncol=2, fontsize='small', handlelength=1.0,
             loc='center left', bbox_to_anchor=(1.01, 0.5))
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('label_path')
  parser.add_argument('datasetspec_path')
  args = parser.parse_args()
  visualize_from_paths(args.label_path, args.datasetspec_path)
