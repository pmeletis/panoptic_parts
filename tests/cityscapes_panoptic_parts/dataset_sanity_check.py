"""
This script reads the original labels of Cityscapes (CO) and compares them against
the Cityscapes-Panoptic-Parts (CPP) labels. For now it validates that the
semantic and instance level of Cityscapes Panoptic Parts (CPP) is equivalent to
original Cityscapes (CO), i.e., sids_iids_CPP == sids_iids_CO.
"""
import sys
assert float(sys.version[:3]) >= 3.6, 'This test uses Python >= 3.6 functionality.'
import os.path as op
import glob

import numpy as np
from PIL import Image

from panoptic_parts.utils.format import decode_uids

# find all label paths
BASEPATH_LABELS_ORIGINAL = 'tests/tests_files/cityscapes/gtFine'
labels_paths_original = glob.glob(op.join(BASEPATH_LABELS_ORIGINAL, 'train', '*', '*_instanceIds.png'))
labels_paths_original.extend(glob.glob(op.join(BASEPATH_LABELS_ORIGINAL, 'val', '*', '*_instanceIds.png')))
print(len(labels_paths_original))
labels_paths_ours = [
    lp.replace('cityscapes', 'cityscapes_panoptic_parts').replace('_instanceIds.png', '_panopticIds.tif')
    for lp in labels_paths_original]
print(len(labels_paths_ours))

# validate labels
for i, (lp_orig, lp_ours) in enumerate(zip(labels_paths_original, labels_paths_ours)):
  print(f"{i+1}/{len(labels_paths_original)}")
  labels_orig = np.array(Image.open(lp_orig), dtype=np.int32)
  labels_ours = np.array(Image.open(lp_ours), dtype=np.int32)

  _, _, _, sids_iids = decode_uids(labels_ours, return_sids_iids=True)
  if not np.all(np.equal(labels_orig, sids_iids)):
    print(lp_orig, lp_ours, sep='\n')
    print(np.unique(labels_orig), print(np.unique(sids_iids)), np.unique(labels_ours), sep='\n')
    breakpoint()
