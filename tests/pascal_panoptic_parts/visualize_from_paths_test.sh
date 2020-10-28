#! /bin/sh
# Change the paths below for your system and run this script from top-level dir as:
# bash tests/pascal_panoptic_parts/visualize_from_paths_test.sh

python -m panoptic_parts.pascal_panoptic_parts.visualize_from_paths \
  tests/tests_files/pascal_panoptic_parts/images/2010_002877.jpg \
  tests/tests_files/pascal_panoptic_parts/labels/2010_002877.tif \
  panoptic_parts/utils/defs/ppp_100.yaml
