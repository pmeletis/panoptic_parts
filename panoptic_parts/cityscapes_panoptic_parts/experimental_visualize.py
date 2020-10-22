"""
Run this script as
`python -m panoptic_parts.cityscapes_panoptic_parts.experimental_visualize <image_path> <label_path>`
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


# prepare SID2COLOR constant needed in experimental_visualize()
# SID2COLOR is a mapping from all possible sids to colors
DEF_PATH = op.join('panoptic_parts', 'utils', 'defs', 'cpp_20.yaml')
with open(DEF_PATH) as fp:
  defs = yaml.load(fp)
SID2COLOR = defs['sid2color']
# add colors for all sids that may exist in labels, but don't have a color from defs
SID2COLOR.update({sid: SID2COLOR[-1]
                  for sid in range(defs['max_sid'])
                  if not (sid in defs['valid_sids'] or sid in SID2COLOR)})


def experimental_visualize(image_path, label_path, experimental_emphasize_instance_boundary=True):
  """
  Visualizes in a pyplot window an image and a label pair from
  provided paths. For reading Pillow is used so all paths and formats
  must be Pillow-compatible.

  Args:
    image_path: an image path provided to Pillow.Image.open
    label_path: a label path provided to Pillow.Image.open
  """
  # reduce resolution for faster execution
  image = Image.open(image_path).resize((1024, 512))
  label = Image.open(label_path).resize((1024, 512), resample=Image.NEAREST)
  uids = np.array(label, dtype=np.int32)

  uids_sem_inst_parts_colored, uids_sem_colored, uids_sem_inst_colored  = \
      experimental_colorize_label(uids,
                                  sid2color=SID2COLOR,
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
  args = parser.parse_args()
  experimental_visualize(args.image_path, args.label_path)
