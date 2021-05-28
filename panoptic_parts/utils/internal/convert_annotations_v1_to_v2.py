import glob
import os
import os.path as op
import sys
from tqdm import tqdm

import numpy as np
from PIL import Image

from panoptic_parts.utils.format import decode_uids, encode_ids


def convert():
  basepath = 'pascal_panoptic_parts/releases/20201704/pascal_panoptic_parts_v1'

  filepaths = glob.glob(op.join(basepath, 'training/*.tif')) + glob.glob(op.join(basepath, 'validation/*.tif'))

  for fp in tqdm(filepaths):
    uids = np.asarray(Image.open(fp), dtype=np.int32)
    # transformation 1 (tvmonitor-unlabeled becomes tvmonitor-frame): {20_XXX, 20_XXX_00} -> 20_XXX_02
    sids, iids, pids, sids_iids, sids_pids =  decode_uids(uids, return_sids_iids=True, return_sids_pids=True)
    pids = np.where(np.logical_and(iids >= 0,
                                   np.logical_or(np.equal(sids_pids, 20), np.equal(sids_pids, 20_00))),
                    2,
                    pids)
    uids = encode_ids(sids, iids, pids)
    # transformation 1 (remove 00): XX_XXX_00 -> XX_XXX
    _, _, pids, sids_iids = decode_uids(uids, return_sids_iids=True)
    uids = np.where(np.logical_and(uids >= 1_000_00, np.equal(pids, 0)),
                    sids_iids,
                    uids)

    path_new = fp.replace('20201704/pascal_panoptic_parts_v1', '20210503/pascal_panoptic_parts_v2')
    assert not op.exists(path_new), f'path {path_new} exists.'
    os.makedirs(op.dirname(path_new), exist_ok=True)
    Image.fromarray(uids, mode='I').save(path_new, format='TIFF', compression='tiff_lzw')


def validate():
  basepath_v1 = 'pascal_panoptic_parts/releases/20201704/pascal_panoptic_parts_v1'
  basepath_v2 = 'pascal_panoptic_parts/releases/20210503/pascal_panoptic_parts_v2'

  filepaths_v1 = glob.glob(op.join(basepath_v1, 'training/*.tif')) + glob.glob(op.join(basepath_v1, 'validation/*.tif'))
  filepaths_v2 = [fp.replace('20201704/pascal_panoptic_parts_v1', '20210503/pascal_panoptic_parts_v2') for fp in filepaths_v1]

  for i, (f1, f2) in enumerate(zip(filepaths_v1, filepaths_v2)):
    l1 = np.asanyarray(Image.open(f1), dtype=np.int32)
    l2 = np.asanyarray(Image.open(f2), dtype=np.int32)
    # if there are differences print the unique tuples with (uid_l1, uid_l2) corresponding
    # to the same spatial position
    cond = l1 != l2
    if np.any(cond):
      uids_tuples = np.unique(np.stack([l1[cond], l2[cond]]), axis=1)
      print(i, *(uids_tuples[:, j] for j in range(uids_tuples.shape[1])))
    else:
      print('No diff.')


if __name__ == '__main__':
  # convert()
  validate()
