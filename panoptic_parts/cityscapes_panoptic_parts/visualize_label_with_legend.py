"""
Run this script as
`python -m panoptic_parts.cityscapes_panoptic_parts.visualize_label_with_legend \
     <label_path> <task_def_path>`
to visualize a Cityscapes-Panoptic-Parts label in all three levels (semantic, instance, parts),
together with a legend including all the colors and uids in that label.
"""

import os.path as op
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml

from panoptic_parts.utils.visualization import (experimental_colorize_label, uid2color)


def visualize_from_paths(label_path, task_def_path):
  """
  Visualizes in a pyplot window a label from the provided path.
  For reading files, Pillow is used, so the path and format
  must be Pillow-compatible. The task definition is used to define colors
  for label ids (see panoptic_parts/utils/defs/template_v1.0.yaml).

  For visualization pixels are colored:
    - semantic-level ids: according to colors defined in task_def
    - semantic-instance-level ids: with random shades of colors defined in task_def
    - semantic-instance-parts-level ids: with a mixture of parula colormap and the shades above
  See panoptic_parts.utils.visualization.uid2color for more information on color generation.

  Args:
    label_path: a label path, will be passed to Pillow.Image.open
    task_def_path: a YAML file path, including keys: `sid2color`, `max_sid`, `valid_sids`
  """
  # sid2color is a mapping from all possible sids to colors
  with open(task_def_path) as fp:
    task_def = yaml.load(fp)
  sid2color = task_def['sid2color']
  # add colors for all sids that may exist in labels, but don't have a color from task_def
  sid2color.update({sid: sid2color[-1]
                    for sid in range(task_def['max_sid'])
                    if not (sid in task_def['valid_sids'] or sid in sid2color)})

  uids = np.array(Image.open(label_path), dtype=np.int32)

  uids_sem_inst_parts_colored = experimental_colorize_label(
      uids, sid2color=sid2color, emphasize_instance_boundaries=True)

  # plot
  _, ax1 = plt.subplots()

  # generate legend, h is a hidden rectangle just to create a legend entry
  handles = []
  handles_text = []
  uids_unique = np.unique(uids)
  uid2color_dict = uid2color(uids_unique, sid2color=sid2color)
  for uid in uids_unique:
    h = plt.Rectangle((0, 0), 1, 1, fc=list(map(lambda x: x/255, uid2color_dict[uid])))
    handles.append(h)
    handles_text.append(str(uid))

  ax1.imshow(uids_sem_inst_parts_colored)
  ax1.set_title('labels colored on semantic, instance, and part levels', fontsize='small')
  ax1.legend(handles, handles_text, ncol=3, fontsize='x-small', handlelength=1.0,
             loc='center left', bbox_to_anchor=(1.01, 0.5))
  plt.tight_layout()
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('label_path')
  parser.add_argument('task_def_path')
  args = parser.parse_args()
  visualize_from_paths(args.label_path, args.task_def_path)
