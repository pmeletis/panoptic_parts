"""
Library for PQ-based evaluation functions.

TODO: add License for copied code
"""
import os
import sys
import os.path as op
import json
import functools
import traceback
import multiprocessing
import copy
from collections import defaultdict

import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

from panoptic_parts.utils.format import decode_uids
from panoptic_parts.utils.utils import _sparse_ids_mapping_to_dense_ids_mapping as ndarray_from_dict


class PQStatCat():
  def __init__(self):
    self.iou = 0.0
    self.tp = 0
    self.fp = 0
    self.fn = 0

  def __iadd__(self, pq_stat_cat):
    self.iou += pq_stat_cat.iou
    self.tp += pq_stat_cat.tp
    self.fp += pq_stat_cat.fp
    self.fn += pq_stat_cat.fn
    return self


class PQStat():
  def __init__(self):
    self.pq_per_cat = defaultdict(PQStatCat)

  def __getitem__(self, i):
    return self.pq_per_cat[i]

  def __iadd__(self, pq_stat):
    for label, pq_stat_cat in pq_stat.pq_per_cat.items():
      self.pq_per_cat[label] += pq_stat_cat
    return self

  def pq_average(self, cat_definition):
    cats_w_parts = []
    cats_no_parts = []
    for i, part_cats in enumerate(cat_definition['cat_def']):
      if len(part_cats['parts_cls']) > 1:
        cats_w_parts.append(i)
      else:
        cats_no_parts.append(i)

    pq, sq, rq, n = 0, 0, 0, 0
    pq_p, sq_p, rq_p, n_p = 0, 0, 0, 0
    pq_np, sq_np, rq_np, n_np = 0, 0, 0, 0
    per_class_results = {}
    for label in range(cat_definition['num_cats']):
      iou = self.pq_per_cat[label].iou
      tp = self.pq_per_cat[label].tp
      fp = self.pq_per_cat[label].fp
      fn = self.pq_per_cat[label].fn
      if tp + fp + fn == 0:
        per_class_results[label] = {'PartPQ': 0.0, 'PartSQ': 0.0, 'PartRQ': 0.0}
        continue
      n += 1
      pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
      sq_class = iou / tp if tp != 0 else 0
      rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
      per_class_results[label] = {'PartPQ': pq_class, 'PartSQ': sq_class, 'PartRQ': rq_class}
      pq += pq_class
      sq += sq_class
      rq += rq_class

      if label in cats_w_parts:
        n_p += 1
        pq_p += pq_class
        sq_p += sq_class
        rq_p += rq_class
      elif label in cats_no_parts:
        n_np += 1
        pq_np += pq_class
        sq_np += sq_class
        rq_np += rq_class

    return [{'PartPQ': pq / n, 'PartSQ': sq / n, 'PartRQ': rq / n, 'n': n},
            {'PartPQ_parts': pq_p / n_p, 'PartSQ_parts': sq_p / n_p, 'PartRQ_parts': rq_p / n_p, 'n_p': n_p},
            {'PartPQ_noparts': pq_np / n_np, 'PartSQ_noparts': sq_np / n_np, 'PartRQ_noparts': rq_np / n_np, 'n_np': n_np}], per_class_results


def prediction_parsing(cat_definition, sem_map, inst_map, part_map):
  '''
  parse the predictions (macro-semantic map, instance map, part-semantic map) into the dict format for evaluation.

  Args:   cat_definition: e.g.
                          {
                              "num_cats": 2,
                              "cat_def":  [{
                                              "sem_cls": [24, 25],
                                              "parts_cls": [1, 2, 3, 4]
                                          },
                                          {
                                              "sem_cls": [26, 27, 28],
                                              "parts_cls": [1, 2, 3, 4, 5]
                                          }]
                          }
          sem_map, inst_map, part_map: 2D numpy arrays, with the same size (H,W)

  Returns: a dict:
          {
              cat #0: {   "num_instances": int,
                          "binary_masks": numpy array with size num_instances*h*w
                          "parts_annotation": numpy array with size num*instances*h*w
                          (for each instance, one layer for mask (binary), one layer for parts segmentation with
                          annotation from 1 to 4 or 5)
                      }
              ,
              cat #1: {
                          ...
                      }
              ,
              ...
          }
  '''

  # shape check
  assert sem_map.shape == inst_map.shape == part_map.shape
  h, w = sem_map.shape

  # map's dtype cannot be uint, since -1 flag is used later
  sem_map = sem_map.astype(np.int32)
  inst_map = inst_map.astype(np.int32)
  part_map = part_map.astype(np.int32)

  meta_dict = {}

  # cat_id is 0, 1, 2, ...,
  for cat_id in range(cat_definition['num_cats']):
    sem_cls = cat_definition['cat_def'][cat_id]['sem_cls']
    parts_cls = cat_definition['cat_def'][cat_id]['parts_cls']

    # empty list for multiple semantic classes for a single category
    binary_masks_list = []
    parts_annotations_list = []

    for sem_idx in sem_cls:
      selected = sem_map == sem_idx
      selected_ins = inst_map.copy()
      selected_ins[np.invert(selected)] = -1
      if -1 in selected_ins:
        idxs, counts = np.unique(selected_ins, return_counts=True)
        # get rid of -1 label stats
        idxs = idxs[1:]
        counts = counts[1:]
      else:
        # only if all the pixels belong to the same semantic classes, then there will be no -1 label
        idxs, counts = np.unique(selected_ins, return_counts=True)

      binary_masks = np.zeros((idxs.shape[0], h, w)).astype(np.int32)
      parts_annotations = np.zeros((idxs.shape[0], h, w)).astype(np.int32)

      # save the masks and part-level annotations
      for i in range(len(idxs)):
        binary_masks[i, :, :] = selected_ins == idxs[i]
        if len(parts_cls) > 1:
          temp_parts = np.zeros((h, w))
          temp_parts[selected_ins == idxs[i]] = part_map[selected_ins == idxs[i]]
          parts_annotations[i, :, :] = temp_parts

      binary_masks_list.append(binary_masks)
      parts_annotations_list.append(parts_annotations)

    binary_masks_per_cat = np.concatenate(binary_masks_list)
    parts_annotations_per_cat = np.concatenate(parts_annotations_list)
    num_instances_per_cat = binary_masks_per_cat.shape[0]

    meta_dict[cat_id] = {'num_instances': num_instances_per_cat,
                         'binary_masks': binary_masks_per_cat,
                         'parts_annotation': parts_annotations_per_cat
                         }

  return meta_dict


def UNUSED_parse_dataset_sid_pid2eval_sid_pid(dataset_sid_pid2eval_sid_pid, experimental_noinfo_id=0):
  """
  Parsing priority, sid_pid is mapped to:
    1. dataset_sid_pid2eval_sid_pid[sid_pid] if it exists, else
    2. dataset_sid_pid2eval_sid_pid[sid] if it exists, else
    3. dataset_sid_pid2eval_sid_pid['DEFAULT'] value

  Returns:
    sid_pid2eval_id: a dense mapping having keys for all possible sid_pid s (0 to 99_99)
      using the provided sparse dataset_sid_pid2eval_sid_pid
  """
  dsp2spe = copy.copy(dataset_sid_pid2eval_sid_pid)
  dsp2spe_keys = dsp2spe.keys()
  dsp2spe_new = dict()
  for k in range(10000):
    if k in dsp2spe_keys:
      dsp2spe_new[k] = dsp2spe[k]
      continue
    sid, pid = (k, None) if k < 100 else divmod(k, 100)
    if sid in dsp2spe_keys:
      dsp2spe_new[k] = dsp2spe[sid]
      continue
    if 'DEFAULT' in dsp2spe_keys:
      dsp2spe_new[k] = dsp2spe['DEFAULT']
      continue
    raise ValueError(f'dataset_sid_pid2eval_sid_pid does not follow the specification rules for key {k}.')
  assert all(v in list(range(10000)) + ['IGNORED'] for v in dsp2spe_new.values())
  # replace ignored sid_pid s with the experimental_noinfo_id
  dsp2spe_new = {k: experimental_noinfo_id if v == 'IGNORED' else v for k, v in dsp2spe_new.items()}
  return dsp2spe_new


def parse_dataset_sid_pid2eval_sid_pid(dataset_sid_pid2eval_sid_pid):
  """
  Parsing priority, sid_pid is mapped to:
    1. dataset_sid_pid2eval_sid_pid[sid_pid] if it exists, else
    2. to the same sid_pid

  Returns:
    sid_pid2eval_id: a dense mapping having keys for all possible sid_pid s (0 to 99_99)
      using the provided sparse dataset_sid_pid2eval_sid_pid
  """
  dsp2spe_new = dict()
  for k in range(10000):
    sid_pid_new = dataset_sid_pid2eval_sid_pid.get(k, k)
    dsp2spe_new[k] = sid_pid_new if sid_pid_new != 'IGNORED' else k
  assert all(v in list(range(10000)) for v in dsp2spe_new.values()), dsp2spe_new.values()
  return dsp2spe_new


def annotation_parsing(spec, sample):
  '''
  parse the numpy encoding defined by dataset definition.

  Args:   
          spec.cat_definition: e.g.
                          {
                              "num_cats": 2,
                              "cat_def":  [{
                                              "sem_cls": [24, 25],
                                              "parts_cls": [1, 2, 3, 4]
                                          },
                                          {
                                              "sem_cls": [26, 27, 28],
                                              "parts_cls": [1, 2, 3, 4, 5]
                                          }]
                          }
          sample: a numpy array with ground truth annotation

  Returns: a dict:
          {
              cat #0: {   "num_instances": int,
                          "binary_masks": numpy array with size num_instances*h*w
                          "parts_annotation": numpy array with size num*instances*h*w
                          (for each instance, one layer for mask (binary), one layer for parts segmentation with
                          annotation from 1 to 4 or 5)
                      }
              ,
              cat #1: {
                          ...
                      }
              ,
              ...
          }
  '''

  h, w = sample.shape

  noinfo_id = 0
  sem_map, inst_map, part_map, sids_pids = decode_uids(sample,
                                                       return_sids_pids=True,
                                                       experimental_noinfo_id=noinfo_id,
                                                       experimental_dataset_spec=spec._dspec)
  sem_map = sem_map.astype(np.int32)
  inst_map = inst_map.astype(np.int32)
  part_map = part_map.astype(np.int32)

  # transform the pids to the pids of the eval_spec, according to dataset_sid_pid2eval_sid_pid,
  # this happends only if dataset_sid_pid2eval_sid_pid is not the identity mapping (k!=v), which
  # this applies only to PPP eval_spec as parts are grouped, while CPP does not group parts
  if any(k != v if v != 'IGNORED' else False for k, v in spec.dataset_sid_pid2eval_sid_pid.items()):
    dsp2esp = parse_dataset_sid_pid2eval_sid_pid(spec.dataset_sid_pid2eval_sid_pid)
    dsp2esp = ndarray_from_dict(dsp2esp, -10**6, length=10000) # -10**6: a random big number
    sids_pids = dsp2esp[sids_pids]
    assert not np.any(np.equal(sids_pids, -10**6)), 'sanity check'
    pids = np.where(sids_pids >= 1_00, sids_pids % 100, noinfo_id)
    # TODO(panos): for now only the pids are mapped, the sids are assumed to be the identical between
    #   the dataset (sem_map) and the eval_spec, so assign only new pids to part_map
    part_map = pids

  meta_dict = {}

  ignore_map = np.zeros((h, w), dtype=np.int32)

  # cat_id is 0, 1, 2, ...,
  cat_definition = spec.cat_definition
  for cat_id in range(cat_definition['num_cats']):
    sem_cls = cat_definition['cat_def'][cat_id]['sem_cls']
    parts_cls = cat_definition['cat_def'][cat_id]['parts_cls']

    # empty list for multiple semantic classes for a single category
    binary_masks_list = []
    parts_annotations_list = []

    for sem_idx in sem_cls:
      selected = sem_map == sem_idx
      selected_ins = inst_map.copy()
      selected_ins[np.invert(selected)] = -1
      if -1 in selected_ins:
        idxs = np.unique(selected_ins)
        # get rid of -1 label stat
        idxs = idxs[1:]
      else:
        # only used if all the pixels belong to the same semantic classes, then there will be no -1 label
        idxs = np.unique(selected_ins)

      binary_masks = np.zeros((idxs.shape[0], h, w)).astype(np.int32)
      parts_annotations = np.zeros((idxs.shape[0], h, w)).astype(np.int32)

      # write the masks and part-level annotations
      for i in range(len(idxs)):
        binary_masks[i, :, :] = selected_ins == idxs[i]
        if len(parts_cls) > 1:
          temp_parts = np.zeros((h, w)).astype(np.int32)
          temp_parts[selected_ins == idxs[i]] = part_map[selected_ins == idxs[i]]
          parts_annotations[i, :, :] = temp_parts

      # Some segments for scene-classes l_parts do not have part annotations (only the background label 0)
      # We cannot apply part-level evaluation to these segments, so we delete them and denote them as crowd
      if len(parts_cls) > 1:
        delete_idx = []
        for i in range(idxs.shape[0]):
          temp_binary_msk = binary_masks[i, :, :]
          temp_parts_anno = parts_annotations[i, :, :]
          part_elements = np.unique(temp_parts_anno[temp_binary_msk > 0.5])
          if part_elements.size == 1 and 0 in part_elements:
            delete_idx.append(i)
            ignore_map[temp_binary_msk > 0.5] = sem_idx
        binary_masks = np.delete(binary_masks, delete_idx, 0)
        parts_annotations = np.delete(parts_annotations, delete_idx, 0)

      binary_masks_list.append(binary_masks)
      parts_annotations_list.append(parts_annotations)

    binary_masks_per_cat = np.concatenate(binary_masks_list)
    parts_annotations_per_cat = np.concatenate(parts_annotations_list)
    num_instances_per_cat = binary_masks_per_cat.shape[0]

    meta_dict[cat_id] = {'num_instances': num_instances_per_cat,
                         'binary_masks': binary_masks_per_cat,
                         'parts_annotation': parts_annotations_per_cat
                         }

  return meta_dict, ignore_map


def generate_ignore_info_tiff(part_panoptic_gt, eval_spec):
  ignore_img = np.zeros_like(part_panoptic_gt).astype(np.uint8)

  # TODO(daan): currently, this is applied to the original part_panoptic tifs, and not to the format on which we wish to evaluate.
  # TODO(daan): this is not an issue now, but can be when using different eval_sids wrt the dataset_sids, it will be problematic

  # get sid iid pid
  sid, _, _, sid_iid = decode_uids(part_panoptic_gt, return_sids_iids=True, experimental_dataset_spec=eval_spec._dspec)

  # if sid not in l_total: set to 255 (void)
  sid_void = np.logical_not(np.isin(sid, eval_spec.eval_sid_total))
  ignore_img[sid_void] = 255

  # if sid_iid < 1000 and sid in l_things, set to crowd and store sid
  no_iid = sid_iid < 1000
  things = np.isin(sid, eval_spec.eval_sid_things)
  crowd = np.logical_and(no_iid, things)

  ignore_img[crowd] = sid_iid[crowd]

  return ignore_img


# will be deleted in the final refactoring pass
def UNUSED_generate_ignore_info(panoptic_dict, panoptic_ann_img, image_id, void=0):
  # Create empty ignore_img and ignore_dict
  ignore_img = np.zeros_like(panoptic_ann_img).astype(np.uint8)
  ignore_dict = dict()

  # Get panoptic segmentation in the correct format
  pan_ann_format = panoptic_ann_img[..., 0] + panoptic_ann_img[..., 1] * 256 + panoptic_ann_img[..., 2] * 256 * 256

  # Store overall void info in ignore_img and ignore_dict
  overall_void = pan_ann_format == void
  ignore_img[overall_void] = 255
  ignore_dict['255'] = 255

  # Retrieve annotation corresponding to image_id
  annotation_dict = dict()
  for annotation in panoptic_dict['annotations']:
    if annotation['image_id'] == image_id:
      annotation_dict = annotation

  if len(annotation_dict) == 0:
    raise KeyError('ImageID is not present in the panoptic annotation dict.')

  # Find crowd annotations and add them to ignore_img and ignore_dict
  for inst_annotation in annotation_dict['segments_info']:
    if inst_annotation['iscrowd'] == 1:
      crowd_instance_id = inst_annotation['id']
      category_id = inst_annotation['category_id']
      crowd_mask = pan_ann_format == crowd_instance_id
      ignore_img[crowd_mask] = category_id
      ignore_dict[str(category_id)] = category_id

  return ignore_img[:, :, 0], ignore_dict


def ignore_img_parsing(sample, cat_definition):
  '''
  parse the ignore_img, which contains crowd (with semantics id) and void region (255)

  Args:   sample: a numpy array
          cat_definition: e.g.
                          {
                              "num_cats": 2,
                              "cat_def":  [{
                                              "sem_cls": [24, 25],
                                              "parts_cls": [1, 2, 3, 4]
                                          },
                                          {
                                              "sem_cls": [26, 27, 28],
                                              "parts_cls": [1, 2, 3, 4, 5]
                                          }]
                          }
  Returns: a dict:
          {
              cat #0: {
                          "binary_masks": numpy array with size num_instances*h*w
                         }
              ,
              cat #1: {
                          ...
                      }
              ,
              ...
          }
  '''
  h, w = sample.shape

  meta_dict = {}

  # cat_id is 0, 1, 2, ...
  for cat_id in range(cat_definition['num_cats']):
    sem_cls = cat_definition['cat_def'][cat_id]['sem_cls']

    binary_masks_per_cat_void = np.zeros((h, w), dtype=np.uint8)
    binary_masks_per_cat_crowd = np.zeros((h, w), dtype=np.uint8)

    for sem_idx in sem_cls:
      binary_masks_per_cat_crowd[sample == sem_idx] = 1

    binary_masks_per_cat_void[sample == 255] = 1

    meta_dict[cat_id] = {'void_masks': binary_masks_per_cat_void,
                         'crowd_masks': binary_masks_per_cat_crowd}

  return meta_dict


def pq_part(pred_meta, gt_meta, crowd_meta, cat_definition, pred_void_label):
  '''

  Args: three meta_dict of the prediction and ground truth, and crowd_instances with definition:
          {
              cat #0: {   "num_instances": int,
                          "binary_masks": numpy array with size num_instances*h*w
                          "parts_annotation": numpy array with size num*instances*h*w
                          (for each instance, one layer for mask (binary), one layer for parts segmentation with
                          annotation from 1 to 4 or 5)
                      }
              ,
              cat #1: {
                          ...
                      }
              ,
              ...
          }
      , cat_definition: e.g.
                          {
                              "num_cats": 2,
                              "cat_def":  [{
                                              "sem_cls": [24, 25],
                                              "parts_cls": [1, 2, 3, 4]
                                          },
                                          {
                                              "sem_cls": [26, 27, 28],
                                              "parts_cls": [1, 2, 3, 4, 5]
                                          }]
                          }

  Returns: an instance PQStat
  '''

  pq_stat = PQStat()

  for cat_id in range(cat_definition['num_cats']):
    pred_ins_dict = pred_meta[cat_id]
    gt_ins_dict = gt_meta[cat_id]
    crowd_ins_dict = crowd_meta[cat_id]

    num_ins_pred = pred_ins_dict['num_instances']
    masks_pred = pred_ins_dict['binary_masks'].astype(np.int32)
    parts_pred = pred_ins_dict['parts_annotation'].astype(np.int32)
    # Set void label
    if pred_void_label != 255:
      parts_pred[parts_pred == pred_void_label] = 255

    num_ins_gt = gt_ins_dict['num_instances']
    masks_gt = gt_ins_dict['binary_masks'].astype(np.int32)
    parts_gt = gt_ins_dict['parts_annotation'].astype(np.int32)

    masks_crowd = crowd_ins_dict['crowd_masks'].astype(np.int32)
    masks_void = crowd_ins_dict['void_masks'].astype(np.int32)
    masks_void_and_crowd = np.logical_or(masks_void, masks_crowd)

    # If a GT segment is a 'crowd' segment (things segment without instance label), do not include in evaluation
    for i in range(num_ins_gt):
      temp_gt_mask = np.logical_and(masks_gt[i, :, :], np.logical_not(masks_crowd))
      if np.sum(temp_gt_mask) == 0:
        num_ins_gt -= 1
        masks_gt = np.delete(masks_gt, i, 0)
        parts_gt = np.delete(parts_gt, i, 0)
        break

    # Loop over the remaining GT segments, find matches, and calculate (part-level) iou
    unmatched_pred = list(range(num_ins_pred))
    for i in range(num_ins_gt):
      temp_gt_mask = np.logical_and(masks_gt[i, :, :], np.logical_not(masks_crowd))
      temp_gt_parts = parts_gt[i, :, :]

      # Loop over all predicted segments to find a match with the T
      for j in range(num_ins_pred):
        if j not in unmatched_pred: continue
        temp_pred_mask = masks_pred[j, :, :]
        temp_pred_parts = parts_pred[j, :, :]

        # Calculate the instance-level IOU between the GT and the predicted segment
        mask_inter_sum = np.sum(np.logical_and(temp_gt_mask, temp_pred_mask))
        mask_union_sum = np.sum(np.logical_or(temp_gt_mask, temp_pred_mask)) - np.sum(
          np.logical_and(masks_void, temp_pred_mask))
        mask_iou = mask_inter_sum / mask_union_sum

        # If the instance-level IOU between ground truth and prediction is larger than 0.5, there is a match
        # In this case, it is a true positive, and the IOU should be evaluated
        if mask_iou > 0.5:
          unmatched_pred.remove(j)
          # For segments of classes with parts, the IOU is the multi-class part-level IOU
          if len(cat_definition['cat_def'][cat_id]['parts_cls']) > 1:
            # The regions in the GT segment for which no parts are defined, are not used for IOU evaluation
            msk_not_defined_in_gt = np.logical_and(temp_gt_parts == 0, temp_gt_mask)
            # The void an crowd regions in the GT are also not used for evaluation
            msk_ignore = np.logical_or(masks_void_and_crowd, msk_not_defined_in_gt)
            msk_evaluated = np.logical_not(msk_ignore)
            # Calculate the confusion matrix for the region that is evaluated (the entire image excl. msk_evaluated)
            cm = confusion_matrix(temp_gt_parts[msk_evaluated].reshape(-1), temp_pred_parts[msk_evaluated].reshape(-1))
            # If there is an 'unknown' prediction in the predicted segment, void_in_pred is True
            void_in_pred = 255 in np.unique(temp_pred_parts[msk_evaluated])
            if cm.size != 0:
              # Calculate IOUs for the part classes (including background and 'unknown' predictions)
              inter = np.diagonal(cm)
              union = np.sum(cm, 0) + np.sum(cm, 1) - np.diagonal(cm)
              ious = inter / np.where(union > 0, union, np.ones_like(union))
              # If there is an 'unknown' prediction in the segment, these pixels should not count as FPs
              if void_in_pred:
                ious = ious[:-1]
              mean_iou = np.mean(ious)
            else:
              raise Exception('empty CM')
            pq_stat[cat_id].tp += 1
            pq_stat[cat_id].iou += mean_iou
          else:
            # For segments of classes without parts, the IOU is the binary instance-level IOU
            pq_stat[cat_id].tp += 1
            pq_stat[cat_id].iou += mask_iou
          break

    # For the remaining unmatched predicted segments, add them as false positives
    # if they are not matched with the void/crowd regions
    for j in range(num_ins_pred):
      if j not in unmatched_pred: continue
      temp_pred_mask = masks_pred[j, :, :]
      mask_inter_sum = np.sum(np.logical_and(masks_void_and_crowd, temp_pred_mask))
      mask_pred_sum = np.sum(temp_pred_mask)
      if mask_inter_sum / mask_pred_sum <= 0.5:
        pq_stat[cat_id].fp += 1

    # The amount of false positives is the total amount of GT segments minus the GT segments that were matched (TPs)
    pq_stat[cat_id].fn = num_ins_gt - pq_stat[cat_id].tp

  return pq_stat


# The decorator is used to prints an error thrown inside process
def get_traceback(f):
  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    try:
      return f(*args, **kwargs)
    except Exception as e:
      print('Caught exception in worker thread:')
      traceback.print_exc()
      raise e

  return wrapper


@get_traceback
def evaluate_single_core(proc_id, fn_pairs, pred_reader_fn, spec):
  # Initialize PartPQ statistics
  pq_stats_split = PQStat()
  cat_definition = spec.cat_definition


  counter = 0
  print(f'core {proc_id}: {counter}/{len(fn_pairs)}')
  # Loop over all predictions
  for fn_pair in fn_pairs:
    counter += 1
    if counter % (len(fn_pairs) // 5) == 0: # print progress 5 times in total 
      print(f'core {proc_id}: {counter}/{len(fn_pairs)}')
    gt_pan_part_file = fn_pair[0]
    pred_file = fn_pair[1]

    # PartPQ eval starts here
    # Load GT annotation file for this image and parse to usable dictionary
    part_gt_sample = np.array(Image.open(gt_pan_part_file)).astype(np.int32)
    part_gt_dict, ignore_img_extra = annotation_parsing(spec, part_gt_sample)

    # Load prediction for this image
    pan_classes, pan_inst_ids, parts_output = pred_reader_fn(pred_file)

    # Parse predictions into usable dictionary
    part_pred_dict = prediction_parsing(cat_definition, pan_classes,
                                        pan_inst_ids, parts_output)

    # Generate data on crowd and void regions
    ignore_img = generate_ignore_info_tiff(part_gt_sample, spec)

    # add removed segments to crowd
    ignore_img[ignore_img_extra != 0] = ignore_img_extra[ignore_img_extra != 0]

    # Parse information about crowd and void segments to usable dictionary
    crowd_dict = ignore_img_parsing(ignore_img, cat_definition)

    # calculate PartPQ per image
    temp_pq_part = pq_part(part_pred_dict, part_gt_dict, crowd_dict, cat_definition, spec.ignore_label)

    pq_stats_split += temp_pq_part

  return pq_stats_split


def evaluate_PartPQ_multicore(spec,
                              filepaths_pairs,
                              pred_reader_fn,
                              cpu_num=round(multiprocessing.cpu_count()/2)):

  assert len(filepaths_pairs) >= cpu_num

  cat_definition = spec.cat_definition

  fn_splits = np.array_split(filepaths_pairs, cpu_num)
  print("Number of cores to be used: {}, images per core: {}".format(cpu_num, len(fn_splits[0])))
  workers = multiprocessing.Pool(processes=cpu_num)
  processes = []
  for proc_id, fn_split in enumerate(fn_splits):
      p = workers.apply_async(evaluate_single_core, (proc_id, fn_split, pred_reader_fn, spec))
      processes.append(p)

  pq_stats_global = PQStat()
  for p in processes:
    pq_stats_global += p.get()

  results = pq_stats_global.pq_average(cat_definition)

  return results
