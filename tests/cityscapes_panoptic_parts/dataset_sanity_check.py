"""
This script reads the original labels of Cityscapes (CO) and compares them against
the Cityscapes-Panoptic-Parts (CPP) labels. It verifies that the semantic and instance
level labels of Cityscapes Panoptic Parts (CPP) are equivalent to
original Cityscapes (CO), i.e., sids_iids_CPP == sids_iids_CO.
"""
import sys
assert float(sys.version[:3]) >= 3.6, 'This test uses Python >= 3.6 functionality.'
import os.path as op
import glob
import multiprocessing

import numpy as np
from PIL import Image

from panoptic_parts.utils.format import decode_uids

# find all label paths
BASEPATH_LABELS_ORIGINAL = 'tests/tests_files/cityscapes/gtFine'
labels_paths_original = glob.glob(op.join(BASEPATH_LABELS_ORIGINAL, 'train', '*', '*_instanceIds.png'))
labels_paths_original.extend(glob.glob(op.join(BASEPATH_LABELS_ORIGINAL, 'val', '*', '*_instanceIds.png')))
print(len(labels_paths_original))
labels_paths_ours = [
    lp.replace('cityscapes/gtFine', 'cityscapes_panoptic_parts/gtFine_v2').replace('_instanceIds.png', 'PanopticParts.tif')
    for lp in labels_paths_original]
print(len(labels_paths_ours))

def _sids_iids_are_maintained(inpts):
  lp_orig, lp_ours = inpts
  labels_orig = np.asarray(Image.open(lp_orig), dtype=np.int32)
  labels_ours = np.asarray(Image.open(lp_ours), dtype=np.int32)
  _, _, _, sids_iids = decode_uids(labels_ours, return_sids_iids=True)
  returns = np.all(np.equal(labels_orig, sids_iids))
  # if not returns:
  #   print(lp_orig, lp_ours, sep='\n')
  #   print(np.unique(labels_orig), print(np.unique(sids_iids)), np.unique(labels_ours), sep='\n')
  return returns

# validate labels
with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
  maintained_bools =[mb for mb in pool.imap_unordered(
      _sids_iids_are_maintained, zip(labels_paths_original, labels_paths_ours), chunksize=10)]

print(len(maintained_bools), 'files were verified.')
assert all(maintained_bools), 'some sids_iids are not the same'
