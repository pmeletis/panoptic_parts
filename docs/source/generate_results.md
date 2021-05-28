# Generate part-aware panoptic segmentation results

Here, we provide a guide for generating Part-aware Panoptic Segmentation (PPS) results as in in our CVPR paper. 

## Prepare EvalSpec and dataset information
Before generating the Part-aware Panoptic Segmentation (PPS) results, you have to specify the dataset you wish to do this for. This consists of two parts:
1. Defining what category definition you wish to use, by using the EvalSpec.
2. Defining which images your dataset contains, and what their properties are.

### EvalSpec
In the `EvalSpec`, we define the following properties
* The classes that are to be evaluated, both on scene-level and part-level
* The split between _things_ and _stuff_ categories, and _parts_ and _no-parts_ categories
* The category definition and numbering that we expect for the predictions.

For the datasets that we define and use in our paper, we provide the `EvalSpec` that we use:
* [ppq_cpp_19_23_cvpr21_default_evalspec.yaml](https://github.com/pmeletis/panoptic_parts/blob/v2.0/panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_default_evalspec.yaml): Cityscapes Panoptic Parts default (parts not grouped)
* [ppq_cpp_19_23_cvpr21_grouped_evalspec.yaml](https://github.com/pmeletis/panoptic_parts/blob/v2.0/panoptic_parts/specs/eval_specs/ppq_cpp_19_23_cvpr21_grouped_evalspec.yaml): Cityscapes Panoptic Parts default (similar parts grouped)
* [ppq_ppp_59_57_cvpr21_default_evalspec.yaml](https://github.com/pmeletis/panoptic_parts/blob/v2.0/panoptic_parts/specs/eval_specs/ppq_ppp_59_57_cvpr21_default_evalspec.yaml): PASCAL Panoptic Parts default


Using these `EvalSpec` definitions, we map the label definition for the raw ground-truth to the definition that we use for evaluation.

**NOTE**: This `EvalSpec` also determines how our merging code expects the predictions. If you do not use the merging code, we expect you to deliver the predictions directly in the 3-channel format, as explained [here](evaluate_results.md).

Examples for CPP default:
* In `eval_sid2_scene_label`, we list the evaluation ids for the scene-level classes and their labels.
  * Following this, the prediction label for `road` is `7`, `car` is `26`, etc.
* In `eval_pid_flat2scene_part_class`, we list the flat evaluation ids for part-level classes as we expect it in a part segmentation output:
  * Each part has a unique id (unless part grouping is used)
  * Following this, the prediction label for `person-head` is `2`, `rider-head` is `6`, etc.
  
You can adjust the `EvalSpec` according to your needs, so you can adjust the mappings and the label definition you use for evaluation.

### Dataset information
To run the merging scripts, we need to know what images are in a given split of a dataset. 
Therefore, for each split (e.g., Cityscapes Panoptic Parts val), we create a json file called `images.json`.

This `images.json` follows the format also used in the [panopticapi](https://github.com/cocodataset/panopticapi), and contains of:
* A dictionary with the key `'images'`, for which the value is:
  * A list of dictionaries with image information. For each image, the dictionary contains:
    * `file_name`: the file name of the RGB image (NOT the ground-truth file).
    * `image_id`: a unique identifier for each image.
    * `height` and `width`: the pixel dimensions of the RGB image (and ground-truth file).

NOTE: the `image_id` defined here, should be unique, and should be used in the names of all prediction files, as explained later.

To generate the `images.json` file for Cityscapes, run the following script from the main `panoptic_parts` directory:

```shell
python -m panoptic_parts.prepare_data \
    $DATASET_DIR \
    $OUTPUT_DIR \
    $DATASET
```
where

- `$DATASET_DIR`: path to the PPS ground-truths file for the data split (e.g. '~/Cityscapes/gtFinePanopticParts_trainval/gtFinePanopticParts/val')
- `$OUTPUT_DIR`: directory where the images.json file will be stored
- `$DATASET`: dataset name ('Cityscapes' or 'Pascal')

## Get results for subtasks
To generate Part-aware Panoptic Segmentation (PPS) predictions, we need to merge panoptic segmentation and part segmentation predictions. Here, we explain how to retrieve and format the predictions on these subtasks, before merging to PPS.

### Panoptic segmentation
There are two options to get panoptic segmentation results:
1. Merge semantic segmentation and instance segmentation predictions. See below how to format and merge these predictions.
2. Do predictions with network that outputs panoptic segmentation results directly.

In the case of option 2, the output needs to be stored in the format as defined for the [COCO dataset](https://cocodataset.org/#format-results):
1. A folder with PNG files storing the ids for all predicted segments.
2. A single .json file storing the semantic information for all images.

For more details on the format, check [here](https://cocodataset.org/#format-results).

**Example Cityscapes Panoptic Parts**: for a baseline in our paper, we generate results for Cityscapes using the provided ResNet-50 model from the [UPSNet repository](https://github.com/uber-research/UPSNet).

### Semantic segmentation
For semantic segmentation:
* For each image, the semantic segmentation prediction should be stored as a single PNG
* Shape: the shape of the corresponding image, i.e., `2048 x 1024` for Cityscapes.
* Each pixel has one value: the scene-level `category_id`, as defined in the `EvalSpec`.
* Name of the files: should include the unique `image_id` as defined in `images.json`.


**Example Cityscapes Panoptic Parts**: for a baseline in our paper, we generate results for Cityscapes using the [provided Xception-65 model](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) from the official [DeepLabv3+ repository](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md).


### Instance segmentation
For instance segmentation, we accept two formats:
1. COCO format (as defined [here](https://cocodataset.org/#format-data).)
2. Cityscapes format (as defined in the comments for [Instance Level Semantic Labeling here](https://github.com/mcordts/cityscapesScripts#evaluation).)
  

For the **COCO format**, we expect:
* A single .json file per image 
* Each json file named as `image_id.json`, with the `image_id` as defined in `images.json`.
* The category id in the json file should be the scene-level id as defined in the `EvalSpec`.

For the **Cityscapes format**, we expect:
* A single .txt file per image, containing per-instance info on each line:\
```relPathPrediction1 labelIDPrediction1 confidencePrediction1 ```

* The category id, (`labelIDPrediction` in the example), should be the scene-level id as defined in the `EvalSpec`.
* The name of each .txt file contains the `image_id` as defined in `images.json`.
* A singe .png containing with a mask prediction for each individual detected instance.
* See the [official Cityscapes repository](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py) for more details.


When merging with semantic segmentation to panoptic segmentation, indicate which instance segmentation format ('COCO' or 'Cityscapes') is used.

**Example Cityscapes Panoptic Parts**: for a baseline in our paper, we generate results for Cityscapes using the official [provided ResNet-50-FPN Mask R-CNN](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) model from the [Detectron2 repository](https://github.com/facebookresearch/detectron2).


### Part segmentation
For part segmentation, we expect predictions in the same format as semantic segmentation:
* For each image, the part segmentation prediction should be stored as a single PNG
* Shape: the shape of the corresponding image, i.e., `2048 x 1024` for Cityscapes.
* Each pixel has one value: the _flat_ part-level `category_id`, as defined in the `EvalSpec`.
* Name of the files: should include the unique `image_id` as defined in `images.json`.


**Example Cityscapes Panoptic Parts**: for a baseline in our paper, we have trained a [BSANet](http://cvteam.net/projects/2019/multiclass-part.html) model with ResNet-101 backbone on our part annotations for the Cityscapes dataset. [These can be downloaded here](INSERT LINK!). **TODO: INSERT LINK**


## Merge instance and semantic segmentation to panoptic segmentation
To use the merging script, you need [pycocotools](https://github.com/cocodataset/cocoapi) and [panopticapi](https://github.com/cocodataset/panopticapi).

These can be installed through pip:
```
pip install pycocotools
pip install git+https://github.com/cocodataset/panopticapi.git
```

To merge to panoptic, run the command below. This generates the images and JSON file with the panoptic segmentation predictions in the format [as defined here](https://cocodataset.org/#format-results), and saves them in `$OUTPUT_DIR`.

From the main `panoptic_parts` directory, run:

```shell
python -m panoptic_parts.merging.merge_to_panoptic \
    $EVAL_SPEC_PATH \
    $INST_PRED_PATH \
    $SEM_PRED_PATH \
    $OUTPUT_DIR \
    $IMAGES_JSON \
    --instseg_format=$INSTSEG_FORMAT

```
where
- `$EVAL_SPEC_PATH`: path to the EvalSpec
- `$INST_PRED_PATH`: path where the instance segmentation predictions are stored (a directory when instseg_format='Cityscapes', a JSON file when instseg_format='COCO')
- `$SEM_PRED_PATH`: path where the semantic segmentation predictions are stored
- `$OUTPUT_DIR`: directory where you wish to store the panoptic segmentation predictions
- `$IMAGES_JSON`: the json file with a list of images and corresponding image ids
- `$INSTSEG_FORMAT`: instance segmentation encoding format, i.e., 'COCO' or 'Cityscapes' (optional, default is 'COCO')


## Merge panoptic and part segmentation to PPS
To merge panoptic segmentation and part segmentation to the Part-aware Panoptic Segmentation (PPS) format, run the code below. 
It stores the PPS predictions as a 3-channel PNG in shape `[height x width x 3]`, where the 3 channels encode the `[scene_category_id, scene_instance_id, part_category_id]`.

From the main `panoptic_parts` directory, run:

```shell
python -m panoptic_parts.merging.merge_to_pps \
    $EVAL_SPEC_PATH \
    $PANOPTIC_PRED_DIR \
    $PANOPTIC_PRED_JSON \
    $PART_PRED_PATH \
    $IMAGES_JSON \
    $OUTPUT_DIR
```

where

- `$EVAL_SPEC_PATH`: path to the EvalSpec
- `$PANOPTIC_PRED_DIR`: directory where the panoptic segmentation predictions (png files) are stored
- `$PANOPTIC_PRED_JSON`: path to the .json file with the panoptic segmentation predictions
- `$PART_PRED_PATH`: directory where the part predictions are stored
- `$IMAGES_JSON`: the json file with a list of images and corresponding image ids
- `$OUTPUT_DIR`: directory where you wish to store the part-aware panoptic segmentation predictions

## Evaluate results
We provide a step-by-step guide for evaluating PPS results. [Click here](EVALUATE_RESULTS.md).


## References and useful links
- [Cityscapes dataset](https://www.cityscapes-dataset.com/)
- [Cityscapes scripts](https://github.com/mcordts/cityscapesScripts) 
- [COCO dataset](https://cocodataset.org/#home)
- [COCO API](https://github.com/cocodataset/cocoapi)
- [COCO Panoptic API](https://github.com/cocodataset/panopticapi)
- [Pascal VOC 2010 dataset](http://host.robots.ox.ac.uk/pascal/VOC/)

