## Installation

The code can be used by installing the requirements and cloning the repository (at least Python 3.7 is needed). A pip package will be available soon.

```shell
git clone https://github.com/pmeletis/panoptic_parts.git
cd panoptic_parts
pip install -r requirements.txt
```

This repository is tested with the following configuration (Linux system):

* Required
  * Python >= 3.7
  * Numpy >= 1.15
  * Pillow >= 8.0
  * SciPy >= 1.4
  * ruamel.yaml >= 0.15

* Optional
  * Tensorflow >= 2.4.0 (for label format handling)
  * Pytorch >= 1.7.0 (for label format handling)
  * Matplotlib >= 3.3.0 (for visualization scripts)
  * panopticapi (for evaluation)
  * pycocotools (for merging)
