.. Cityscapes-Panoptic-Parts and PASCAL-Panoptic-Parts for Scene Understanding documentation master file, created by
   sphinx-quickstart on Thu Jan 28 11:43:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Cityscapes-Panoptic-Parts and PASCAL-Panoptic-Parts for Scene Understanding's documentation!
=======================================================================================================

.. toctree::
   :hidden:

   Home Page <self>
   Label Format <label_format>
   API Reference <_autosummary/panoptic_parts>
   Contact <contact>

This repository contains code and tools for reading, processing, and visualizing *Cityscapes-Panoptic-Parts* and *PASCAL-Panoptic-Parts* datasets. We created these datasets by extending two established datasets for image scene understanding, namely `Cityscapes <https://github.com/mcordts/cityscapesScripts>`_ and `PASCAL <http://host.robots.ox.ac.uk/pascal/VOC/voc2010/>`_ datasets.

Citation
--------

Detailed description of the datasets and various statistics are presented in our technical report in `arxiv <https://arxiv.org/abs/2004.07944>`_. Please cite us if you find our work useful and you use it for your research:

.. code-block:: bibtex

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

Examples
--------

Cityscapes-Panoptic-Parts
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 
      .. image:: ../../readme/aachen_000012_000019_leftImg8bit.jpg
         :target: ../../readme/aachen_000012_000019_leftImg8bit.jpg
         :alt: Image
     
     - 
      .. image:: ../../readme/aachen_000012_000019_uids_pids_colored.png
         :target: ../../readme/aachen_000012_000019_uids_pids_colored.png
         :alt: Image
     
   * - 
      .. image:: ../../readme/frankfurt_000001_011835_leftImg8bit.jpg
         :target: ../../readme/frankfurt_000001_011835_leftImg8bit.jpg
         :alt: Image
     
     - 
      .. image:: ../../readme/frankfurt_000001_011835_uids_pids_colored.png
         :target: ../../readme/frankfurt_000001_011835_uids_pids_colored.png
         :alt: Image
     


PASCAL-Panoptic-Parts
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - 
      .. image:: ../../readme/2008_000393.jpg
         :target: ../../readme/2008_000393.jpg
         :alt: Image
     
     - 
      .. image:: ../../readme/2008_000393_colored.png
         :target: ../../readme/2008_000393_colored.png
         :alt: Image
     
     - 
      .. image:: ../../readme/2008_000716.jpg
         :target: ../../readme/2008_000716.jpg
         :alt: Image
     
     - 
      .. image:: ../../readme/2008_000716_colored.png
         :target: ../../readme/2008_000716_colored.png
         :alt: Image
     
   * - 
      .. image:: ../../readme/2008_007456.jpg
         :target: ../../readme/2008_007456.jpg
         :alt: Image
     
     - 
      .. image:: ../../readme/2008_007456_colored_repainted.png
         :target: ../../readme/2008_007456_colored_repainted.png
         :alt: Image
     
     - 
      .. image:: ../../readme/2010_002356.jpg
         :target: ../../readme/2010_002356.jpg
         :alt: Image
      
     - 
      .. image:: ../../readme/2010_002356_colored.png
         :target: ../../readme/2010_002356_colored.png
         :alt: Image
     

Installation
------------

Use the following command to install the requirements:

.. code-block:: bash

    pip3 install -r requirements.txt

The following packages are optional for usage:

* Tensorflow >= 2.4.0 (for label format handling)
* Pytorch >= 1.7.0 (for label format handling)
* Matplotlib >= 3.3.0 (for visualization scripts)
* panopticapi (for PASCAL visualization script)

Code usage
----------

We provide a public, backwards compatible API, which allows easier bug fixes and functionality updates. We suggest that the users update their local clone of this repository frequently by pulling the master branch. The list can be found here: :doc:`Public API <_autosummary/panoptic_parts>`.

All functions and arguments named with the preffix 'experimental\_' or with an '\_' do not
belong to the stable API and may change.

Hierarchical format and labels encoding
---------------------------------------

We encode three levels of labels: semantic, instance, and parts in a single image-like file. The hierarchical panoptic encoding of the labels is explained here: :doc:`Label format </label_format>`. Labels for both datasets follow this format.

Ground Truth usage cases
------------------------

We provide for each image a single (image-like) ground truth file encoding semantic-, instance-, and parts- levels annotations. Our compact :doc:`Label format </label_format>` together with `decode\_uids <../../panoptic_parts/utils/format.py>`_ function enable easy decoding of the labels for various image understanding tasks including:

.. code-block:: Python

   # labels: Python int, or np.ndarray, or tf.Tensor, or torch.tensor

   # Semantic Segmentation
   semantic_ids, _, _ = decode_uids(labels)

   # Instance Segmentation
   semantic_ids, instance_ids, _ = decode_uids(labels)

   # Panoptic Segmentation
   _, _, _, semantic_instance_ids = decode_uids(labels, return_sids_iids=True)

   # Parts Segmentation / Parts Parsing
   _, _, _, semantic_parts_ids = decode_uids(labels, return_sids_pids=True)

   # Instance-level Parts Parsing
   semantic_ids, instance_ids, parts_ids = decode_uids(labels)

   # Parts-level Panoptic Segmentation
   _, _, _, semantic_instance_ids, semantic_parts_ids = decode_uids(labels, return_sids_iids=True, return_sids_pids=True)
