## Installation

The code can be installed from the PyPI and requires at least Python 3.7. It is recommended to install it in a Python virtual environment.

```shell
pip install panoptic_parts
```

Some functionality requires extra packages to be installed, e.g. evaluation scripts (tqdm) or Pytorch/Tensorflow (torch/tensorflow). These can be installed separately or by downloading the `optional.txt` file from this repo and running the following command in the virtual environment:

```shell
pip install -r optional.txt
```

After installation you can use the package as:

```python
import panoptic_parts as pp

print(pp.VERSION)
```

There are three scripts defined as entry points by the package:

```shell
pp_merge_to_panoptic <args>
pp_merge_to_pps <args>
pp_visualize_label_with_legend <args>
```
