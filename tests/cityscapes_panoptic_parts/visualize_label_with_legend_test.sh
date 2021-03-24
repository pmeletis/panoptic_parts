#! /bin/sh
# Change the paths below for your system and run this script from top-level dir as:
# bash tests/cityscapes_panoptic_parts/visualize_label_with_legend_test.sh

python -m panoptic_parts.cityscapes_panoptic_parts.visualize_label_with_legend \
  tests/tests_files/gtFinePanopticParts/val/munster/munster_000080_000019_gtFinePanopticParts.tif \
  panoptic_parts/utils/defs/cpp_20.yaml
