"""
Run this script as
`python -m pascal_panoptic_parts.experimental_visualize <image_path> <label_path>`
to visualize a Pascal-Panoptic-Parts image and label pair in the following
3 levels: semantic, semantic-instance, semantic-instance-parts.
"""
import argparse
import os.path as op
import json
from copy import deepcopy

from scipy import ndimage
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from panopticapi.utils import IdGenerator

from panoptic_parts.utils.format import decode_uids
from panoptic_parts.utils.utils import color_map

def prepare_categories(def_path):
    # prepare some constants needed in visualize()
    with open(def_path, 'r') as json_file:
        prob_def = json.load(json_file)
    sid_pid2parts_cid = deepcopy(prob_def['sid_pid2parts_cid'])
    sid_pid2parts_cid = {int(k): v for k, v in sid_pid2parts_cid.items()}
    categories_list = deepcopy(prob_def['categories'])
    categories_list_parts = deepcopy(prob_def['categories'])
    num_semantic_classes = len(categories_list_parts) + 1

    # Load the sid_pid to parts encoding and add them into categories as distinct thing classes with their own sids (to differentiate slightly in colors for different instances).
    for sid_pid, parts_cid in sid_pid2parts_cid.items():
        if parts_cid == 0:
            # For part-level unlabeled classes, we regard it as a semantic stuff class to have consistent colors around instances.
            sid_pid2parts_cid[sid_pid] = int(sid_pid // 100)
        else:
            # For regular part-level classes, we regard it as a new semantic thing class for visualization.
            sid_pid2parts_cid[sid_pid] += num_semantic_classes - 1

    # Register the part-level classes and initialize base color for these "sid_pid thing" classes.
    part_colors = color_map(N=(num_semantic_classes + len(sid_pid2parts_cid))).tolist()[num_semantic_classes:]
    for ind, (sid_pid, parts_cid) in enumerate(sid_pid2parts_cid.items()):
        if parts_cid != 0:
            categories_list_parts.append({"id": parts_cid, "name": "", "color": part_colors[ind], "supercategory": "object", "isthing": 1})

    categories_list.append({"id": 0, "name": "", "color": [0, 0, 0], "supercategory": "background", "isthing": 0})
    categories_list_parts.append({"id": 0, "name": "", "color": [0, 0, 0], "supercategory": "background", "isthing": 0})
    return ({category['id']: category for category in categories_list},
            {category['id']: category for category in categories_list_parts},
            sid_pid2parts_cid)


DEF_PATH = op.join('utils', 'defs', 'ppp_100classes.json')
CATEGORIES, CATEGORIES_PARTS, SID_PID2PARTS_CID = prepare_categories(DEF_PATH)


def visualize(image_path, label_path):
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

    # Prepare canvases and decode the labels.
    image = np.array(Image.open(image_path), dtype=np.uint8)
    label = np.array(Image.open(label_path), dtype=np.int32)
    uids_unique_org = np.unique(label)
    semantic_segmentation = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    instance_segmentation = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    parts_segmentation = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    sids, iids, _ = decode_uids(label)

    # Color at the semantic level.
    color_generator = IdGenerator(CATEGORIES)
    for sid in np.unique(sids):
        mask = np.equal(sids, sid)
        color = CATEGORIES[sid]['color']
        semantic_segmentation[mask] = color

    # Color at the semantic and instance level and find the instance-level boundaries.
    sids_only = np.where(iids < 0, sids, np.zeros_like(iids))
    for sid in np.unique(sids_only):
        mask = np.equal(sids_only, sid)
        color = color_generator.get_color(sid)
        instance_segmentation[mask] = color

    sid_iids = np.where(iids >= 0, sids * 10**3 + iids, np.zeros_like(iids))
    boundaries = np.full(sid_iids.shape, False)
    for sid_iid in np.unique(sid_iids):
        if sid_iid != 0:
            mask = np.equal(sid_iids, sid_iid)
            color = color_generator.get_color(sid_iid // 1000)
            instance_segmentation[mask] = color
            boundary_horizon = ndimage.sobel(mask, 0)
            boundary_vertical = ndimage.sobel(mask, 1)
            boundaries = np.logical_or(np.hypot(boundary_horizon, boundary_vertical), boundaries)

    # Color at the part level.
    # Conver the original labels into the form for visualization with IdGenerator.
    for uid in uids_unique_org:
        # If uid is sid or sid_iid, encode them as they are.
        if uid <= 99_999:
            sid_iid = uid
        # If uid is sid_iid_pid, map sid_pid to its corresponding sid and create new label as sid_iid.
        else:
            sid, iid, pid = decode_uids(uid)
            sid_pid = sid * 10**2 + pid
            if sid_pid in SID_PID2PARTS_CID:
                sid_iid = SID_PID2PARTS_CID[sid_pid] * 10**3 + iid
            else:
                sid_iid = sid * 10**3 + iid

        label[label == uid] = sid_iid

    color_generator = IdGenerator(CATEGORIES_PARTS)

    for sid_iid in np.unique(label):
        # If sid_iid is in the format of sid , use sid for color generation (things and stuff classes differentiated by IdGenerator inherently).
        if sid_iid <= 99:
            id_ = sid_iid
        # If sid_iid is in the format of sid_iid, get sid.
        else:
            id_ = sid_iid // 1000
        mask = label == sid_iid
        color = color_generator.get_color(id_)
        parts_segmentation[mask] = color

    # Depict boundaries.
    instance_segmentation[boundaries] = [255, 255, 255]
    parts_segmentation[boundaries] = [255, 255, 255]

    # plot
    # initialize figure for plotting
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    # for ax in axes:
    #   ax.set_axis_off()
    ax1.imshow(image)
    ax1.set_title('image')
    ax2.imshow(semantic_segmentation)
    ax2.set_title('labels colored on semantic level')
    ax3.imshow(instance_segmentation)
    ax3.set_title('labels colored on semantic and instance levels')
    ax4.imshow(parts_segmentation)
    ax4.set_title('labels colored on semantic, instance, and parts levels')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('label_path')
    args = parser.parse_args()
    visualize(args.image_path, args.label_path)
