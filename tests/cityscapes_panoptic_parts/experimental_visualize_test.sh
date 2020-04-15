#! /bin/sh

python -m cityscapes_panoptic_parts.visualize \
  tests/test_files/cityscapes_panoptic_parts/leftImg8bit/train/aachen/aachen_000012_000019_leftImg8bit.png \
  tests/test_files/cityscapes_panoptic_parts/gtFine/train/aachen/aachen_000012_000019_gtFine_panopticIds.tif
