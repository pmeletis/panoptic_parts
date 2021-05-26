"""
Run this script as
`python -m panoptic_parts.cityscapes_panoptic_parts.visualize_from_paths \
     <image_path> <label_path> <task_def_path>`
to visualize a Cityscapes-Panoptic-Parts image and label pair in the following
3 levels: semantic, semantic-instance, semantic-instance-parts.
"""

import os.path as op
import json
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import yaml

from panoptic_parts.utils.visualization import experimental_colorize_label


def visualize_from_paths(image_path, label_path, task_def_path):
  """
  Visualizes in a pyplot window an image and a label pair from
  provided paths. For reading files, Pillow is used so all paths and formats
  must be Pillow-compatible. The task definition is used to define colors
  for label ids (see panoptic_parts/utils/defs/template_v1.0.yaml).

  For visualization pixels are colored:
    - semantic-level ids: according to colors defined in task_def
    - semantic-instance-level ids: with random shades of colors defined in task_def
    - semantic-instance-parts-level ids: with a mixture of parula colormap and the shades above
  See panoptic_parts.utils.visualization.uid2color for more information on color generation.

  Args:
    image_path: an image path, will be passed to Pillow.Image.open
    label_path: a label path, will be passed to Pillow.Image.open
    task_def_path: a YAML file path, including keys: `sid2color`, `max_sid`, `valid_sids`
  """
  # sid2color is a mapping from all possible sids to colors
  with open(task_def_path) as fp:
    task_def = yaml.load(fp, Loader=yaml.Loader)
  sid2color = task_def['sid2color']
  # add colors for all sids that may exist in labels, but don't have a color from task_def
  sid2color.update({sid: sid2color[-1]
                    for sid in range(task_def['max_sid'])
                    if not (sid in task_def['valid_sids'] or sid in sid2color)})

  # reduce resolution for faster execution
  image = Image.open(image_path).resize((1024, 512))
  label = Image.open(label_path).resize((1024, 512), resample=Image.NEAREST)
  uids = np.array(label, dtype=np.int32)

  uids_sem_inst_parts_colored, uids_sem_colored, uids_sem_inst_colored  = \
      experimental_colorize_label(uids,
                                  sid2color=sid2color,
                                  return_sem=True,
                                  return_sem_inst=True,
                                  emphasize_instance_boundaries=True)

  # plot
  # initialize figure for plotting
  _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
  # for ax in axes:
  #   ax.set_axis_off()
  ax1.imshow(image)
  ax1.set_title('image')
  ax2.imshow(uids_sem_colored)
  ax2.set_title('labels colored on semantic level')
  ax3.imshow(uids_sem_inst_colored)
  ax3.set_title('labels colored on semantic and instance levels')
  ax4.imshow(uids_sem_inst_parts_colored)
  ax4.set_title('labels colored on semantic, instance, and parts levels')
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('image_path')
  parser.add_argument('label_path')
  parser.add_argument('task_def_path')
  args = parser.parse_args()
  visualize_from_paths(args.image_path, args.label_path, args.task_def_path)
