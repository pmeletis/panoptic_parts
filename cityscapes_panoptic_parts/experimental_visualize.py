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
from scipy import ndimage

from utils.utils import uids_lids2uids_cids, safe_write, _sparse_ids_mapping_to_dense_ids_mapping
from utils.format import decode_uids
from utils.visualization import _colorize_uids, uid2color

DEF_PATH = op.join('utils', 'defs', 'cityscapes_default_20classes.json')

# prepare some constants needed in visualize()
with open(DEF_PATH, 'r') as fp:
  prob_def = json.load(fp)
LIDS2CIDS = prob_def['lids2cids']
# replace voids (-1) with max+1
LIDS2CIDS = np.array([m if m!=-1 else max(LIDS2CIDS)+1 for m in LIDS2CIDS], dtype=np.int32)
CIDS2COLORS = {cid: tuple(map(int, color)) for cid, color in enumerate(prob_def['cids2colors'])}

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

  image = Image.open(image_path)
  uids_with_lids = np.array(Image.open(label_path), dtype=np.int32)

  # uids according to our hierarchical panoptic format
  uids_with_cids = uids_lids2uids_cids(uids_with_lids, LIDS2CIDS)

  # We want to visualize on all three levels so we need all the uids levels
  # and we do it here for all levels together so we call uid2color once to have
  # same shades across subfigures per plot for easier comparison
  sids, iids, _, sids_iids = decode_uids(uids_with_cids, experimental_return_sids_iids=True)
  ids_all_levels_unique = np.unique(np.stack([sids, sids_iids, uids_with_cids]))
  uid_2_color = uid2color(list(map(int, ids_all_levels_unique)), CIDS2COLORS)
  palette = _sparse_ids_mapping_to_dense_ids_mapping(uid_2_color, (0, 0, 0), dtype=np.uint8)

  # using numpy advanced indexing (gathering) a color from the (Ncolors, 3)-shaped palette
  # is chosen for each sid, sid_iid, and uid
  uids_sem_colored = palette[sids]
  uids_sem_inst_colored = palette[sids_iids]
  uids_sem_inst_parts_colored = palette[uids_with_cids]

  # add boundaries
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
