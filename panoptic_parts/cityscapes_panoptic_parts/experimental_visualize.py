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
from scipy import ndimage
import yaml

from panoptic_parts.utils.utils import (
    uids_lids2uids_cids, safe_write, _sparse_ids_mapping_to_dense_ids_mapping)
from panoptic_parts.utils.format import decode_uids
from panoptic_parts.utils.visualization import uid2color


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
  assert op.exists(image_path)
  assert op.exists(label_path)
  # reduce resolution for faster execution
  image = Image.open(image_path).resize((1024, 512))
  label = Image.open(label_path).resize((1024, 512), resample=Image.NEAREST)
  uids = np.array(label, dtype=np.int32)

  # We visualize labels on three levels: semantic, semantic-instance, semantic-instance-parts.
  # We want to colorize same instances with the same shades across subfigures for easier comparison
  # so we create ids_all_levels_unique and call uid2color() once to achieve that.
  # sids, iids, sids_iids shapes: (height, width)
  sids, iids, _, sids_iids = decode_uids(uids, return_sids_iids=True)
  ids_all_levels_unique = np.unique(np.stack([sids, sids_iids, uids]))
  uid2color_dict = uid2color(ids_all_levels_unique, sid2color=SID2COLOR)

  # We colorize ids using numpy advanced indexing (gathering). This needs an array palette, thus we
  # convert the dictionary uid2color_dict to a "continuous" array with shape (Ncolors, 3) and
  # values in range [0, 255] (RGB).
  # uids_*_colored shapes: (height, width, 3)
  palette = _sparse_ids_mapping_to_dense_ids_mapping(uid2color_dict, (0, 0, 0), dtype=np.uint8)
  uids_sem_colored = palette[sids]
  uids_sem_inst_colored = palette[sids_iids]
  uids_sem_inst_parts_colored = palette[uids]

  # optionally add boundaries to the colorized labels uids_*_colored
  edge_option = 'sobel' # or 'erosion'
  if experimental_emphasize_instance_boundary:
    # TODO(panos): simplify this algorithm
    # create per-instance binary masks
    iids_unique = np.unique(iids)
    boundaries = np.full(iids.shape, False)
    edges = np.full(iids.shape, False)
    for iid in iids_unique:
      if 0 <= iid <= 999:
        iid_mask = np.equal(iids, iid)
        if edge_option == 'sobel':
          edge_horizont = ndimage.sobel(iid_mask, 0)
          edge_vertical = ndimage.sobel(iid_mask, 1)
          edges = np.logical_or(np.hypot(edge_horizont, edge_vertical), edges)
        elif edge_option == 'erosion':
          boundary = np.logical_xor(iid_mask,
                                    ndimage.binary_erosion(iid_mask, structure=np.ones((4, 4))))
          boundaries = np.logical_or(boundaries, boundary)

    if edge_option == 'sobel':
      boundaries_image = np.uint8(edges)[..., np.newaxis] * np.uint8([[[255, 255, 255]]])
    elif edge_option == 'erosion':
      boundaries_image = np.uint8(boundaries)[..., np.newaxis] * np.uint8([[[255, 255, 255]]])

    uids_sem_inst_colored = np.where(boundaries_image,
                                     boundaries_image,
                                     uids_sem_inst_colored)
    uids_sem_inst_parts_colored = np.where(boundaries_image,
                                           boundaries_image,
                                           uids_sem_inst_parts_colored)

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
