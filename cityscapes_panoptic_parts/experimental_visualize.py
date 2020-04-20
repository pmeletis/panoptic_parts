"""
Run this script as
`python -m cityscapes_panoptic_parts.experimental_visualize <image_path> <label_path>`
to visualize a Cityscapes-Panoptic-Parts image and label pair in the following
3 levels: semantic, semantic-instance, semantic-instance-parts.
"""

import os.path as op
import json
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils.utils import uids_lids2uids_cids, safe_write
from utils.visualization import _colorize_uids

DEF_PATH = op.join('utils', 'defs', 'cityscapes_default_20classes.json')

# prepare some constants needed in visualize()
with open(DEF_PATH, 'r') as fp:
  prob_def = json.load(fp)
LIDS2CIDS = prob_def['lids2cids']
# replace voids (-1) with max+1
LIDS2CIDS = np.array([m if m!=-1 else max(LIDS2CIDS)+1 for m in LIDS2CIDS], dtype=np.int32)
CIDS2COLORS = np.array(prob_def['cids2colors'], dtype=np.uint8)

def experimental_visualize(image_path, label_path):
  """
  Visualizes in a pyplot window an image and a label pair from
  provided paths. For reading Pillow is used so all paths and formats
  must be Pillow-compatible.

  Args:
    image_path: an image path provided to Pillow.Image.open
    label_path: a label path provided to Pillow.Image.open
  """
  assert op.exists(image_path)
  assert op.exists(label_path)

  image = Image.open(image_path)
  uids_with_lids = np.array(Image.open(label_path), dtype=np.int32)

  # uids according to our hierarchical panoptic format
  uids_with_cids = uids_lids2uids_cids(uids_with_lids, LIDS2CIDS)
  uids_colored = _colorize_uids(
      uids_with_cids, {cid: color for cid, color in enumerate(CIDS2COLORS)}, True)
  uids_sem_colored, uids_sem_inst_colored, uids_sem_inst_parts_colored = uids_colored

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
