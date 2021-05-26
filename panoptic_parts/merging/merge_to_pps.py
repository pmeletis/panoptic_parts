import numpy as np
import os
import json
import argparse

from tqdm import tqdm
from PIL import Image

from panoptic_parts.specs.merge_eval_spec import PartPQEvalSpec
from panoptic_parts.utils.utils import get_filenames_in_dir, find_filename_in_list


def _prepare_mappings(sid_pid2part_seg_label, void):
  # Get the maximum amount of part_seg labels
  num_part_seg_labels = np.max(
    list(sid_pid2part_seg_label.values()))

  sids2part_seg_ids = dict()
  for class_key in sid_pid2part_seg_label.keys():
    class_id = class_key // 100
    if class_id in sids2part_seg_ids.keys():
      if sid_pid2part_seg_label[class_key] not in sids2part_seg_ids[class_id]:
        sids2part_seg_ids[class_id].append(sid_pid2part_seg_label[class_key])
      else:
        raise ValueError(
          'A part seg id can only be shared between different semantic classes, not within a single semantic class.')
    else:
      sids2part_seg_ids[class_id] = [sid_pid2part_seg_label[class_key]]

  sids2pids_eval = dict()
  for class_key in sid_pid2part_seg_label.keys():
    class_id = class_key // 100
    if class_id in sids2pids_eval.keys():
      if class_key % 100 not in sids2pids_eval[class_id]:
        sids2pids_eval[class_id].append(class_key % 100)
    else:
      sids2pids_eval[class_id] = [class_key % 100]

  part_seg_ids2eval_pids_per_sid = dict()
  for class_key in sids2part_seg_ids.keys():
    tmp = np.ones(num_part_seg_labels + 1, np.uint8) * void
    tmp[sids2part_seg_ids[class_key]] = sids2pids_eval[class_key]
    part_seg_ids2eval_pids_per_sid[class_key] = tmp

  return sids2part_seg_ids, part_seg_ids2eval_pids_per_sid


def _create_categories_list(eval_spec):
  category_list = list()

  for eval_id in eval_spec.eval_sid2scene_label.keys():
    category_dict = dict()
    category_dict['id'] = eval_id
    category_dict['name'] = eval_spec.eval_sid2scene_label[eval_id]
    # TODO: make function in eval_spec to get (eval_sid2scene_color) functionality
    category_dict['color'] = eval_spec.dataset_spec.sid2scene_color[eval_id]
    if eval_id in eval_spec.eval_sid_things:
      category_dict['isthing'] = 1
    else:
      category_dict['isthing'] = 0

    category_list.append(category_dict)

  return category_list

def merge(eval_spec_path,
          panoptic_pred_dir,
          panoptic_pred_json,
          part_pred_path,
          images_json,
          output_dir):
  """
  :param eval_spec_path: path to the EvalSpec
  :param panoptic_pred_dir: directory where the panoptic segmentation predictions (png files) are stored
  :param panoptic_pred_json: path to the .json file with the panoptic segmentation predictions
  :param part_pred_path: directory where the part predictions are stored
  :param images_json: the json file with a list of images and corresponding image ids
  :param output_dir: directory where you wish to store the part-aware panoptic segmentation predictions

  :return:
  """
  eval_spec = PartPQEvalSpec(eval_spec_path)

  # If the output directory does not exist, create it
  if not os.path.exists(output_dir):
    print("Creating output directory at {}".format(output_dir))
    os.mkdir(output_dir)

  # Get category information from EvalSpec
  categories_list = _create_categories_list(eval_spec)
  categories_json = os.path.join(output_dir, 'categories.json')
  with open(categories_json, 'w') as fp:
    json.dump(categories_list, fp)

  # Get the sid_pid -> part_seg mapping from the EvalSpec
  sid_pid2part_seg_label = eval_spec.eval_sid_pid2eval_pid_flat

  # Get the void label from the EvalSpec definition
  void = eval_spec.ignore_label

  # Load list of images in data split
  with open(images_json, 'r') as fp:
    images_dict = json.load(fp)
  images_list = images_dict['images']

  # Load list of panoptic predictions
  with open(panoptic_pred_json, 'r') as fp:
    panoptic_dict = json.load(fp)

  # Prepare the mappings from predictions to evaluation and vice versa
  sids2part_seg_ids, part_seg_ids2eval_pids_per_sid = _prepare_mappings(sid_pid2part_seg_label, void)

  # Load panoptic annotations
  annotations = panoptic_dict['annotations']
  annotations_id_list = [annotation['image_id'] for annotation in annotations]

  # Get filenames in directory with part segmentation predictions
  fn_partseg = get_filenames_in_dir(part_pred_path)

  print("Merging panoptic and part predictions to PPS, and saving...")
  for image_info in tqdm(images_list):
    image_id = image_info['id']

    # Because the panopticapi converts strings to integers when possible, we have to check two cases
    if image_id in annotations_id_list:
      annotation_index = annotations_id_list.index(image_id)
    elif int(image_id) in annotations_id_list:
      annotation_index = annotations_id_list.index(int(image_id))
    else:
      raise FileNotFoundError('No panoptic prediction found for image id {}'.format(image_id))

    annotation = annotations[annotation_index]
    file_name = annotation['file_name']

    # Load and decode panoptic predictions
    pred_pan = np.array(Image.open(os.path.join(panoptic_pred_dir, file_name)))
    pred_pan_flat = pred_pan[..., 0] + pred_pan[..., 1] * 256 + pred_pan[..., 2] * 256 ** 2
    h, w = pred_pan.shape[0], pred_pan.shape[1]

    # Load part predictions
    f_partseg = find_filename_in_list(image_id, fn_partseg, 'part segmentation')
    pred_part = np.array(Image.open(f_partseg))

    class_canvas = np.ones((h, w), dtype=np.int32) * void
    inst_canvas = np.zeros((h, w), dtype=np.int32)
    # TODO(daan): check whether we can also set part_canvas init to 255
    part_canvas = np.zeros((h, w), dtype=np.int32)

    segment_count_per_cat = dict()

    # Loop over all predicted panoptic segments
    for segment in annotation['segments_info']:
      segment_id = segment['id']
      cat_id = segment['category_id']

      # Increase the segment count per category
      if cat_id in segment_count_per_cat.keys():
        segment_count_per_cat[cat_id] += 1
      else:
        segment_count_per_cat[cat_id] = 1

      # Check whether there are not too many segments to store in a PNG w/ dtype np.uint8
      if segment_count_per_cat[cat_id] > 255:
        raise ValueError('More than 255 instances for category_id > {}. This is currently not yet supported.'.format(cat_id))

      mask = pred_pan_flat == segment_id

      # Loop over all scene-level categories
      if cat_id in eval_spec.eval_sid_parts:
        # If category has parts
        # Check what pids are possible for the sid
        plausible_parts = sids2part_seg_ids[cat_id]
        plausible_parts_mask = np.isin(pred_part, plausible_parts)

        # Get the mapping from part_seg ids to evaluation pids, given the sid
        part_seg_ids2eval_pids = part_seg_ids2eval_pids_per_sid[cat_id]
        part_canvas[mask] = void

        # Convert the part seg ids to the desired evaluation pids, and store them in the tensor with part labels
        part_canvas[np.logical_and(mask, plausible_parts_mask)] = part_seg_ids2eval_pids[
          pred_part[np.logical_and(mask, plausible_parts_mask)]]

        # Store the category id and instance id in the respective tensors
        class_canvas[mask] = cat_id
        inst_canvas[mask] = segment_count_per_cat[cat_id]

      else:
        # If category does not have parts
        mask = pred_pan_flat == segment_id

        # Store the category id and instance id in the respective tensors
        class_canvas[mask] = cat_id
        inst_canvas[mask] = segment_count_per_cat[cat_id]
        # Store a dummy part id
        part_canvas[mask] = 1

    pred_pan_part = np.stack([class_canvas, inst_canvas, part_canvas], axis=2)
    img_pan_part = Image.fromarray(pred_pan_part.astype(np.uint8))
    img_pan_part.save(os.path.join(output_dir, file_name))

  # Remove categories json file that was necessary for merging
  os.remove(categories_json)

  print("Merging finished.")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Merges panoptic and part segmentation predictions to part-aware panoptic segmentation results."
  )

  parser.add_argument('eval_spec_path', type=str,
                      help="path to the EvalSpec")
  parser.add_argument('panoptic_pred_dir', type=str,
                      help="directory where the panoptic segmentation predictions (png files) are stored")
  parser.add_argument('panoptic_pred_json', type=str,
                      help="path to the .json file with the panoptic segmentation predictions")
  parser.add_argument('part_pred_path', type=str,
                      help="directory where the part predictions are stored")
  parser.add_argument('images_json', type=str,
                      help="the json file with a list of images and corresponding image ids")
  parser.add_argument('output_dir', type=str,
                      help="directory where you wish to store the part-aware panoptic segmentation predictions")
  args = parser.parse_args()

  merge(args.eval_spec_path,
        args.panoptic_pred_dir,
        args.panoptic_pred_json,
        args.part_pred_path,
        args.images_json,
        args.output_dir)