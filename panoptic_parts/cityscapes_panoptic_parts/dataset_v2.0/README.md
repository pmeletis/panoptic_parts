
# Cityscapes Panoptic Parts annotations
We have manually annotated 5 scene-level classes with 23 part-level classes from Cityscapes vehicle and human categories.

You can download the dataset from the [Cityscapes Dataset](https://www.cityscapes-dataset.com/login/) website.

Pixels of humans and vehicles (_sids_: 24, 25, 26, 27, or 28) that are not assigned to any part-level class by the annotation team or it is not clearly visible to which part they belong to, have _pid_ = 0 or they maintain their semantic-level or semantic-instance-level labels. From the perspective of semantics the labels `SS_III_00` and `SS_III` are equivalent.

## Human (person (_sid_: 24), rider (_sid_: 25)) pids:

* 0: unlabeled / void
* 1: torso
* 2: head
* 3: arms
* 4: legs

> Note: For human and rider scene classes a _pid_ 5 exists in a minority of ground truth files (~10). This _pid_ is an artefact of data preprocessing. These artefact can be automatically set to void _pid_ 0 (unlabeled part) using the decoding functionality provided in the following snippet:

  ```python
  uids = np.array(Image.open('gt_filepath.tif'), dtype=np.int32)
  dataset_spec = DatasetSpec('cpp_datasetspec.yaml')
  _, _, pids = decode_uids(uids, experimental_dataset_spec=dataset_spec, experimental_correct_range=True)
  ```

## Vehicle (car (_sid_: 26), truck (_sid_: 27), bus (_sid_: 28)) pids:

* 0: unlabeled / void
* 1: windows
* 2: wheels
* 3: lights
* 4: license plate
* 5: chassis

## Contact

Please feel free to contact us for any suggestions or questions:

* Panagiotis Meletis: **p**[DOT]**c**[DOT]**meletis**[AT]**tue.nl**
* Xiaoxiao (Vincent) Wen: **wenxx10**[AT]**gmail.com**
