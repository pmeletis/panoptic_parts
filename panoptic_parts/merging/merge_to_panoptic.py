import os
import argparse
import json

import numpy as np
from PIL import Image
from tqdm import tqdm
from panopticapi import combine_semantic_and_instance_predictions
from pycocotools import mask

from panoptic_parts.specs.eval_spec import PartPQEvalSpec
from panoptic_parts.utils.utils import get_filenames_in_dir, find_filename_in_list


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


def _stuff_segmentation_to_coco(sem_pred_dir, images_list, stuff_labels):
  segmentation_coco_format = list()

  fn_semseg = get_filenames_in_dir(sem_pred_dir)

  for image in tqdm(images_list):
    image_id = image['id']

    f_semseg = find_filename_in_list(image_id, fn_semseg, 'semantic segmentation')
    semseg_map = np.array(Image.open(f_semseg)).astype(np.uint8)

    for label_id in stuff_labels:
      # Create mask and encode it
      label_mask = semseg_map == label_id

      # If the category is not in the prediction, go to the next category
      if not np.any(label_mask):
        continue
      label_mask = np.expand_dims(label_mask, axis=2)
      label_mask = np.asfortranarray(label_mask)
      RLE = mask.encode(label_mask)
      assert len(RLE) == 1
      RLE = RLE[0]

      # When using Python3, we convert the encoding to ascii format, to be serializable as json
      RLE['counts'] = RLE['counts'].decode('ascii')

      # Add encoded data to the list
      segmentation_coco_format.append({'segmentation': RLE,
                                       'image_id': image_id,
                                       'category_id': int(label_id)})

  return segmentation_coco_format


def _instance_cs_to_coco_format(inst_pred_dir, images_list):
  instseg_coco_format = list()

  fn_instseg = get_filenames_in_dir(inst_pred_dir)

  for image in tqdm(images_list):
    image_id = image['id']
    h, w = image['height'], image['width']

    # Find the txt file for the image in the directory with predictions
    f_instseg = find_filename_in_list(image_id, fn_instseg, 'instance segmentation', ext='.txt')

    instseg_masks = list()
    instseg_classes = list()
    instseg_scores = list()
    with open(f_instseg, 'r') as txtfile:
      # For each line in the txt file, load the corresponding image and store the mask, class and score
      for txtline in txtfile:
        inst_mask_file = os.path.join(inst_pred_dir, txtline.split(' ')[0])
        instseg_masks.append(np.array(Image.open(inst_mask_file)) / 255)
        instseg_classes.append(txtline.split(' ')[1])
        instseg_scores.append(txtline.split(' ')[2])
    instseg_masks = np.reshape(np.array(instseg_masks).astype(np.uint8),
                               (-1, h, w))

    # Encode mask as RLE as expected by the COCO format
    RLE = mask.encode(np.asfortranarray(np.transpose(instseg_masks, (1, 2, 0))))
    for i, _ in enumerate(instseg_masks):
      # Store all in a list, as expected by the COCO format

      # When using Python3, we convert the encoding to ascii format, to be serializable as json
      RLE[i]['counts'] = RLE[i]['counts'].decode('ascii')

      instseg_coco_format.append({'segmentation': RLE[i],
                                  'score': float(instseg_scores[i]),
                                  'image_id': image_id,
                                  'category_id': int(instseg_classes[i])})

  return instseg_coco_format


def merge(eval_spec_path,
          inst_pred_path,
          sem_pred_path,
          output_dir,
          images_json,
          instseg_format='COCO'):
  """

  :param eval_spec_path: path to the EvalSpec
  :param inst_pred_path: path where the instance segmentation predictions are stored
      (a directory when instseg_format='Cityscapes', a JSON file when instseg_format='COCO')
  :param sem_pred_path: path where the semantic segmentation predictions are stored
  :param output_dir: directory where you wish to store the panoptic segmentation predictions
  :param images_json: the json file with a list of images and corresponding image ids
  :param instseg_format: instance segmentation encoding format (either 'COCO' or 'Cityscapes')

  :return:
  """

  assert instseg_format in ['Cityscapes', 'COCO'], \
      "instseg_format should be \'Cityscapes\' or \'COCO\'"

  eval_spec = PartPQEvalSpec(eval_spec_path)

  # If the output directory does not exist, create it
  if not os.path.exists(output_dir):
    print("Creating output directory at {}".format(output_dir))
    os.mkdir(output_dir)

  # Load list of images with their properties
  with open(images_json, 'r') as fp:
    images_dict = json.load(fp)

  # Get category information from EvalSpec
  categories_list = _create_categories_list(eval_spec)
  categories_json = os.path.join(output_dir, 'categories.json')
  with open(categories_json, 'w') as fp:
    json.dump(categories_list, fp)

  # Get the list of all images in the dataset (split)
  images_list = images_dict['images']

  print("Loading instance segmentation predictions")
  # Load instance segmentation predictions
  if instseg_format == 'Cityscapes':
    print("Converting instance segmentation predictions from CS to COCO format")
    assert os.path.isdir(inst_pred_path), "When instseg_format = 'Cityscapes', inst_pred_path should be a directory." \
                                          "Currently, inst_pred_path is {}.".format(inst_pred_path)
    # If in Cityscapes format, convert to expected COCO format
    inst_pred_list = _instance_cs_to_coco_format(inst_pred_path, images_list)
  elif instseg_format == 'COCO':
    assert inst_pred_path.endswith('.json'), "When instseg_format = 'COCO', inst_pred_path should be a json file." \
                                             "Currently, inst_pred_path is {}.".format(inst_pred_path)
    # If in COCO format, load the json file into a list
    with open(inst_pred_path, 'r') as fp:
      inst_pred_list = json.load(fp)

  # Load semantic segmentation predictions, filter out stuff classes and convert to COCO format
  print("Loading semantic segmentation predictions, and converting to COCO format")
  stuff_labels = eval_spec.eval_sid_stuff
  sem_pred_list = _stuff_segmentation_to_coco(sem_pred_path, images_list, stuff_labels=stuff_labels)

  instseg_json_file = os.path.join(output_dir, 'inst_pred.json')
  with open(instseg_json_file, 'w') as fp:
    json.dump(inst_pred_list, fp)

  semseg_json_file = os.path.join(output_dir, 'sem_pred.json')
  with open(semseg_json_file, 'w') as fp:
    json.dump(sem_pred_list, fp)

  output_json = os.path.join(output_dir, 'panoptic.json')
  output_dir_files = os.path.join(output_dir, 'panoptic')
  combine_semantic_and_instance_predictions.combine_predictions(semseg_json_file,
                                                                instseg_json_file,
                                                                images_json,
                                                                categories_json,
                                                                output_dir_files,
                                                                output_json,
                                                                confidence_thr=0.5,
                                                                overlap_thr=0.5,
                                                                stuff_area_limit=1024)

  # Remove json files that were necessary for merging
  os.remove(semseg_json_file)
  os.remove(instseg_json_file)
  os.remove(categories_json)

  print("Merging finished.")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Merges semantic and instance segmentation predictions to panoptic segmentation results."
  )

  parser.add_argument('eval_spec_path', type=str,
                      help="path to the EvalSpec")
  parser.add_argument('inst_pred_path', type=str,
                      help="path where the instance segmentation predictions are stored (a directory when instseg_format='Cityscapes', a JSON file when instseg_format='COCO')")
  parser.add_argument('sem_pred_path', type=str,
                      help="path where the semantic segmentation predictions are stored")
  parser.add_argument('output_dir', type=str,
                      help="directory where you wish to store the panoptic segmentation predictions")
  parser.add_argument('images_json', type=str,
                      help="the json file with a list of images and corresponding image ids")
  parser.add_argument('--instseg_format', type=str,
                      help="instance segmentation encoding format (either 'COCO' or 'Cityscapes')", default='COCO')
  args = parser.parse_args()

  merge(args.eval_spec_path,
        args.inst_pred_path,
        args.sem_pred_path,
        args.output_dir,
        args.images_json,
        instseg_format=args.instseg_format)

