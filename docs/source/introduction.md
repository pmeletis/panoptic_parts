# Introduction

This repository contains code and tools for reading, processing, evaluating on, and visualizing Panoptic Parts datasets. Moreover, it contains code for reproducing our CVPR 2021 paper results.

## Datasets

*Cityscapes-Panoptic-Parts* and *PASCAL-Panoptic-Parts* are created by extending two established datasets for image scene understanding, namely [Cityscapes](https://github.com/mcordts/cityscapesScripts "Cityscapes") and [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/ "PASCAL") datasets. Detailed description of the datasets and various statistics are presented in our technical report in [arxiv](https://arxiv.org/abs/2004.07944 "arxiv.org"). The datasets can be downloaded from:

- [Cityscapes Panoptic Parts](https://www.cityscapes-dataset.com/login/)
- [PASCAL Panoptic Parts](https://1drv.ms/u/s!AojlpuGgPtL1bHXfIdeL14IeVhI?e=5tNfET)

## API and code reference

We provide a public, stable API, and various code utilities that are documented [here](https://panoptic-parts.readthedocs.io/en/stable/api_and_code.html).

## Citations

 Please cite us if you find our work useful or you use it in your research:

```bibtex
@inproceedings{degeus2021panopticparts,
    title = {Part-aware Panoptic Segmentation},
    author = {Daan de Geus and Panagiotis Meletis and Chenyang Lu and Xiaoxiao Wen and Gijs Dubbelman},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2021}
}
```

```bibtex
@article{meletis2020panopticparts,
    title = {Cityscapes-Panoptic-Parts and PASCAL-Panoptic-Parts datasets for Scene Understanding},
    author = {Panagiotis Meletis and Xiaoxiao Wen and Chenyang Lu and Daan de Geus and Gijs Dubbelman},
    type = {Technical report},
    institution = {Eindhoven University of Technology},
    date = {16/04/2020},
    url = {https://github.com/tue-mps/panoptic_parts},
    eprint={2004.07944},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!-- This is a workaround for the image rendering problem using Markdown with Sphinx. -->

```eval_rst
.. image:: _static/mps_logo.png
    :target: https://www.tue.nl/en/research/research-groups/signal-processing-systems/mobile-perception-systems-lab/
    :alt: MPS
    :height: 100

.. image:: _static/tue_logo.jpg
    :target: https://www.tue.nl/
    :alt: TU/e
    :height: 100
```
