# Panoptic Parts Datasets

This repository contains code and tools for reading, processing, and visualizing *Cityscapes-Panoptic-Parts* and *PASCAL-Panoptic-Parts* datasets. We created these datasets by extending two established datasets for image scene understanding, namely [Cityscapes](https://github.com/mcordts/cityscapesScripts "Cityscapes") and [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/ "PASCAL") datasets.

Detailed description of the datasets and various statistics are presented in our technical report in [arxiv](https://arxiv.org/abs/2004.07944 "arxiv.org"). Please cite us if you find our work useful and you use it for your research:
```
@report{meletisetal2020panopticparts,
	author = {Meletis Panagiotis, Xiaoxiao Wen, Chenyang Lu, Daan de Geus, Gijs Dubbelman},
	title = {Cityscapes-Panoptic-Parts and PASCAL-Panoptic-Parts datasets for Scene Understanding},
	type = {Technical report},
	institution = {Eindhoven University of Technology},
	date = {16/04/2020},
	url = {https://github.com/tue-mps/panoptic_parts},
}
```

## Cityscapes-Panoptic-Parts

---
![Image](readme/aachen_000012_000019_leftImg8bit.jpg "Image") | ![Image](readme/aachen_000012_000019_uids_pids_colored.png "Image")
---- | ----
![Image](readme/frankfurt_000001_011835_leftImg8bit.jpg "Image") | ![Image](readme/frankfurt_000001_011835_uids_pids_colored.png "Image")
---

## PASCAL-Panoptic-Parts

---
![Image](readme/2008_000393.jpg "Image") | ![Image](readme/2008_000393_colored.png "Image") | ![Image](readme/2008_000716.jpg "Image") | ![Image](readme/2008_000716_colored.png "Image")
---- | ---- | ---- | ----
![Image](readme/2008_007456.jpg "Image") | ![Image](readme/2008_007456_colored_repainted.png "Image") | ![Image](readme/2010_002356.jpg "Image") | ![Image](readme/2010_002356_colored.png "Image")
---

## Requirements

Tested with the following configuration:

* Linux system
* Python >= 3.6
* Tensorflow >= 2.0
* Numpy
* Pillow
* SciPy
* Matplotlib (only for visualization scripts)
* panopticapi (only for PASCAL visualization script)

## Code usage

We provide a public, backwards compatible API, which allows easier bug fixes and functionality updates. We suggest that the users update their local clone of this repository frequently by pulling the master branch. The list can be found here: [Public API](API.md).

All functions and arguments named with the preffix 'experimental_' or with an '_' do not
belong to the stable API and may change.

## Contact

Please feel free to contact us for any suggestions or questions:

* Panagiotis Meletis: **p**[DOT]**c**[DOT]**meletis**[AT]**tue.nl**
* Vincent Wen: **xiaoxiao**[DOT]**wen**[AT]**student.uva.nl**
