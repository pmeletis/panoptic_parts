"""
Run this script as
`python -m panoptic_parts.visualization.visualize_label_with_legend \
     <datasetspec_path> <label_path>`
to visualize a label in all three levels (semantic, instance, parts),
together with a legend including all the colors and uids in that label.
"""
import argparse

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from panoptic_parts.utils.visualization import experimental_colorize_label
from panoptic_parts.utils.format import decode_uids, encode_ids
from panoptic_parts.specs.dataset_spec import DatasetSpec


def visualize_from_paths(datasetspec_path, label_path):
  """
  Visualizes in a pyplot window a label from the provided path.

  For visualization pixels are colored on:
    - semantic-level: according to colors defined in dataspec.sid2scene_color
    - semantic-instance-level: with random shades of colors defined in dataspec.sid2scene_color
    - semantic-instance-parts-level: with a mixture of parula colormap and the shades above
  See panoptic_parts.utils.visualization.uid2color for more information on color generation.

  Args:
    datasetspec_path: a YAML file path, including keys:
      `sid2scene_color`, `scene_class_part_class_from_sid_pid`
    label_path: a label path, will be passed to Pillow.Image.open
  """
  spec = DatasetSpec(datasetspec_path)
  uids = np.array(Image.open(label_path), dtype=np.int32)
  # for PPP, we need to fold groupable parts (see dataset ppp_datasetspec.yaml for more details)
  uids = encode_ids(*decode_uids(uids, experimental_dataset_spec=spec, experimental_correct_range=True))

  uids_sem_inst_parts_colored, uid2color_dct = experimental_colorize_label(
      uids, sid2color=spec.sid2scene_color, emphasize_instance_boundaries=True, return_uid2color=True,
      experimental_deltas=(60, 60, 60), experimental_alpha=0.5)

  # plot
  _, ax1 = plt.subplots()

  # generate legend, h is a hidden rectangle just to create a legend entry
  handles = []
  handles_text = []
  uids_unique = np.unique(uids)
  for uid in uids_unique:
    h = plt.Rectangle((0, 0), 1, 1, fc=list(map(lambda x: x/255, uid2color_dct[uid])))
    handles.append(h)
    _, _, _, sid_pid = decode_uids(uid, return_sids_pids=True)
    scene_class_part_class = spec.scene_class_part_class_from_sid_pid(sid_pid)
    handles_text.append(f'{uid}: {scene_class_part_class}')

  ax1.imshow(uids_sem_inst_parts_colored)
  ax1.set_title('labels colored on semantic, instance, and part levels', fontsize='small')
  ax1.legend(handles, handles_text, ncol=3, fontsize='small', handlelength=1.0,
             loc='center left', bbox_to_anchor=(1.01, 0.5))
  plt.tight_layout()
  plt.show()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('datasetspec_path')
  parser.add_argument('label_path')
  args = parser.parse_args()
  visualize_from_paths(args.datasetspec_path, args.label_path)

  return


if __name__ == "__main__":
  main()
