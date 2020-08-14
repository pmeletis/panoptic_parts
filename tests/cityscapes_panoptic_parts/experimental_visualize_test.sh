#! /bin/sh
# Change the paths below for your system and run this script from top-level dir as:
# bash tests/cityscapes_panoptic_parts/experimental_visualize_test.sh

python -m panoptic_parts.cityscapes_panoptic_parts.experimental_visualize \
  tests/tests_files/cityscapes_panoptic_parts/leftImg8bit/train/aachen/aachen_000012_000019_leftImg8bit.png \
  tests/tests_files/cityscapes_panoptic_parts/gtFine/train/aachen/aachen_000012_000019_gtFine_panopticIds.tif
