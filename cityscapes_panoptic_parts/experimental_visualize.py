"""
Run this script as python -m cityscapes_panoptic_parts.visualize to visualize a Cityscapes
image and label pair provided as arguments.
"""

import os.path as op
import json
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils.utils import uids_lids2uids_cids, safe_write
from utils.visualization import colorize_uids

DEF_PATH = op.join('utils', 'defs', 'cityscapes_default_20classes.json')

# prepare some constants needed in visualize()
with open(DEF_PATH, 'r') as fp:
  prob_def = json.load(fp)
LIDS2CIDS = prob_def['lids2cids']
# replace voids (-1) with max+1
LIDS2CIDS = np.array([m if m!=-1 else max(LIDS2CIDS)+1 for m in LIDS2CIDS], dtype=np.int32)
CIDS2COLORS = np.array(prob_def['cids2colors'], dtype=np.uint8)

def visualize():
  """
  Visualizes in a pyplot window iteratively images and labels from
  image_label_output_paths and optionally writes visualizations to disk.
  For reading and writting Pillow is used so all paths and formats
  must be Pillow-compatible. Visualizations are written in an overwrite-safe
  manner (only if the output path does not exist already).

  Note: for fast visualizations generation during 

  Args:
    image_label_output_paths: a list of tuples:
      (image_path, label_path) or (image_path, label_path, output_path).
  """
  parser = argparse.ArgumentParser()
  parser.add_argument('image_path')
  parser.add_argument('label_path')
  args = parser.parse_args()
  image_path = args.image_path
  label_path = args.label_path
  assert op.exists(image_path)
  assert op.exists(label_path)

  image = Image.open(image_path)
  uids_with_lids = np.array(Image.open(label_path))

  # debugging prints
  # print(image_path, type(uids_with_lids), uids_with_lids.dtype,
  #       f'min: {np.min(uids_with_lids)}', f'max: {np.max(uids_with_lids)}', sep=', ')

  # uids according to our panoptic format
  uids_with_cids = uids_lids2uids_cids(uids_with_lids, LIDS2CIDS)
  uids_cids_colored, uids_iids_colored, uids_pids_colored = colorize_uids(
      uids_with_cids, {cid: color for cid, color in enumerate(CIDS2COLORS)}, True)

  # try:
  #   # uids according to our panoptic format
  #   uids_with_cids = uids_lids2uids_cids(uids_with_lids, LIDS2CIDS)
  #   uids_cids_colored, uids_iids_colored, uids_pids_colored = colorize_uids(
  #       uids_with_cids, {cid: color for cid, color in enumerate(CIDS2COLORS)}, True)
  # except Exception as exc:
  #   print(exc)
  #   print(label_path, 'cannot be visualized.')
  #   return False

  # plot
  # initialize figure for plotting
  _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
  # for ax in axes:
  #   ax.set_axis_off()
  ax1.imshow(image)
  ax1.set_title('image')
  ax2.imshow(uids_cids_colored)
  ax2.set_title('labels colored on semantic level')
  ax3.imshow(uids_iids_colored)
  ax3.set_title('labels colored on semantic and instance levels')
  ax4.imshow(uids_pids_colored)
  ax4.set_title('labels colored on semantic, instance, and parts levels')
  plt.show()


if __name__ == "__main__":
  visualize()
