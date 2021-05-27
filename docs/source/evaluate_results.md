# Evaluate on PartPQ metric

To evaluate on the PartPQ metric, you need to follow three steps:
1. Select or prepare the EvalSpec for your data
2. Prepare the part-aware panoptic segmentation predictions in the correct format
3. Run the evaluation script


## 1. Select EvalSpec
In the `EvalSpec`, we define how we wish to evaluate the dataset. Specifically, we define:
* The classes that are to be evaluated, both on scene-level and part-level
* The split between _things_ and _stuff_ categories, and _parts_ and _no-parts_ categories
* The category definition and numbering that we expect for the predictions.

The `EvalSpec`s have the following filename format:
```
{metric-name}_{dataset-name}_{num-scene-classes}_{num-part-classes}_{specific-setting}_evalspec.yaml
```

For the datasets that we define and use in our paper, we provide the `EvalSpec` that we use:
* [ppq_cpp_19_23_cvpr21_default_evalspec.yaml](../../panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_default_evalspec.yaml): Cityscapes Panoptic Parts default (parts not grouped)
* [ppq_cpp_19_23_cvpr21_grouped_evalspec.yaml](../../panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_grouped_evalspec.yaml): Cityscapes Panoptic Parts default (similar parts grouped)
* [ppq_ppp_59_57_cvpr21_default_evalspec.yaml](../../panoptic_parts/specs/eval_specs/ppq_ppp_59_57_cvpr21_default_evalspec.yaml): PASCAL Panoptic Parts default

## 2. Prepare the predictions
Before we can evaluate the results, you should make sure that the predictions are in the proper format. There are two things to be considered:
1. The correct category ids should be used
2. The data should be encoded and provided in the proper 3-channel PNG format. 


### 2.1. Category ids
The category ids in the prediction -- both for scene classes and part classes -- should be provided as defined in the `EvalSpec`. 

1) For scene-level classes:
   * In `eval_sid2scene_label`, we provide the scene category ids that are used during evaluation, and their corresponding names.
   * In the prediction, these category ids should be used.
    

2) For part-level classes:
    * `eval_sid_parts` is a list of scene categories for which we expect part labels.
    * In `eval_sid_pid2eval_pid_flat`, we provide all the `sid_pid` category combinations that are evaluated.
        * The first part of the `sid_pid` is the scene category id (`sid`), the second is the and part category id (`pid`)
        * To see the corresponding category names for these `sid_pid`, see the mapping to the unique `eval_pid_flat`, and the provided class labels in `eval_pid_flat2scene_part_label`.
    * The `pid` from the `sid_pid`is the part category id that we expect in the predictions.


**Example for CPP default:**
1) As follows from `eval_sid2scene_label`:
   * The scene id for `car` is `26`, and `road` is `7`.
2) As follows from `eval_sid_pid2eval_pid_flat` and `eval_pid_flat2scene_part_label`:
   * The combined `sid_pid` prediction label for `person-head` is `24_02`
   * ==> The part id is `2` (and the scene id is `24`). 
    
### 2.2. 3-channel PNG format
In the evaluation script, we expect the predictions to be encoded as a 3-channel PNG (i.e., `HxWx3`), where the channels should encode:
1. Scene category id
2. Instance id (unique for each instance within a scene category)
3. Part category id

These should be encoded as unsigned integers (`uint8`), and the filename of the PNG should include the filename of the original input image for which the prediction is the result.

For regions where there is no prediction, or regions with `unknown` predictions, the category id should be set to `255`.


## 3. Run evaluation script
To run the evaluation script, you need to have a json file containing information on the images that you wish to evaluate on. [Here](generate_results.md#dataset-information), we describe how to generate this `images.json` using `evaluation/prepare_data.py`.

Run the evaluation script from the top-level `panoptic_parts` directory as:

```shell
python -m panoptic_parts.evaluation.eval_PartPQ \
    $EVAL_SPEC_PATH \
    $GT_PATH \
    $PRED_PATH \
    $IMAGES_JSON \
    --save_dir=$SAVE_DIR
```

where:
 - `$EVAL_SPEC_PATH`: selected evaluation specification from Step 1
 - `$GT_PATH`: directory with ground truth files
 - `$PRED_PATH`: directory with prediction files
 - `$IMAGES_JSON`: the images.json file with a list of images and corresponding image ids
 - `$SAVE_DIR`: a directory to save the json file with results (optional)
 
 For more information on the arguments run `python -m panoptic_parts.evaluation.eval_PartPQ.py -h`.
