import yaml

from panoptic_parts.specs.dataset_spec import DatasetSpec


with open('ppp_20_58_iou_evalspec.yaml') as fd:
  gspec = yaml.load(fd, Loader=yaml.Loader)

dspec = DatasetSpec(gspec['dataset_spec_path'])

with open('ppq_ppp_59_57_evalspec.yaml') as fd:
  espec = yaml.load(fd, Loader=yaml.Loader)



# dataset_sid_pid2eval_sid_pid
###################################################################################################
part_groupings = gspec['part_groupings']
dataset_sid_pid2eval_sid_pid = dict()
for sid_pid, (scene_class, part_class) in dspec.sid_pid2scene_class_part_class.items():
  if sid_pid == 0 or scene_class not in part_groupings.keys():
    continue
  sid = sid_pid // 100
  pid_new = None
  # find the part_class position in the part_groupings dict
  for pid_new_cand, (part_class_new, part_classes_old) in enumerate(part_groupings[scene_class].items(), start=1):
    for part_class_old in part_classes_old:
      if part_class_old == part_class:
        pid_new = pid_new_cand
        break
    else: # ie inner loop DOES NOT break, continue mid loop
      continue
    break # if inner loop breaks, then break mid loop
  else: # ie mid loop DOES NOT break, continue outer loop
    continue
  dataset_sid_pid2eval_sid_pid[sid_pid] = sid * 100 + pid_new

# sanity check
esd2epf = espec['eval_sid_pid2eval_pid_flat']
assert all(v in esd2epf.keys() for v in dataset_sid_pid2eval_sid_pid.values())

# print in a friendly copy-paste way to yaml
sid_prev = 0
for k, v in dataset_sid_pid2eval_sid_pid.items():
  sid_cur = k // 100
  if sid_cur > sid_prev:
    sid_prev = sid_cur
    print('\n  ', end='')
  print('{}_{:02d}'.format(*divmod(k, 100)) + ': ' + '{}_{:02d}'.format(*divmod(v, 100)) + ',', end=' ')
###################################################################################################

# eval_sid2scene_label
###################################################################################################
# eval_sid2dataset_sid = espec['eval_sid2scene_label']
# eval_sid2scene_label = {es: dspec.scene_class_from_sid(ds) for es, ds in eval_sid2dataset_sid.items()}
###################################################################################################

# eval_pid_flat2scene_part_label
###################################################################################################
eval_pid_flat = espec['eval_pid_flat2scene_part_label'].keys()
eval_pid_flat2eval_sid_pid = {v: k for k, v in espec['eval_sid_pid2eval_pid_flat'].items()}
eval_pid_flat2eval_sid_pid[0] = 0

part_groupings['UNLABELED'] = {'UNLABELED': ['UNLABELED']}

eval_pid_flat2scene_part_label = dict()
for k in eval_pid_flat:
  eval_sid_pid = eval_pid_flat2eval_sid_pid[k]
  eval_sid, eval_pid = divmod(eval_sid_pid, 100)
  scene_class = dspec.scene_class_from_sid(eval_sid)
  part_class_new2part_classes_old = {'UNLABELED': ['UNLABELED']}
  part_class_new2part_classes_old.update(part_groupings[scene_class])
  part_class = list(part_class_new2part_classes_old.keys())[eval_pid]
  eval_pid_flat2scene_part_label[k] = f'{scene_class}-{part_class}'
###################################################################################################
breakpoint()
