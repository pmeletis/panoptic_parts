## Installation

The code is tested with the following configuration (Linux system):

* Required
  * Python >= 3.6
  * Numpy >= 1.19.4
  * Pillow >= 8.0.1
  * SciPy >= 1.5.4
  * PyYAML >= 5.3.1

* Optional
  * Tensorflow >= 2.4.0 (for label format handling)
  * Pytorch >= 1.7.0 (for label format handling)
  * Matplotlib >= 3.3.0 (for visualization scripts)
  * panopticapi (for PASCAL visualization script)

Use the following command under the project directory to install the requirements:
```bash
pip3 install -r requirements.txt
```

To install the optional packages, run:

```bash
pip3 install -r optional.txt
```