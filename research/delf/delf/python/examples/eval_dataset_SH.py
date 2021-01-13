
# This file will repeatedly call my new function "match_images_SH()"

# python3 match_images.py \
#   --image_1_path data/oxford5k_images/hertford_000056.jpg \
#   --image_2_path data/oxford5k_images/oxford_000317.jpg \
#   --features_1_path data/oxford5k_features/hertford_000056.delf \
#   --features_2_path data/oxford5k_features/oxford_000317.delf \
#   --output_image matched_images.png

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import matplotlib
# Needed before pyplot import for matplotlib to work properly.
matplotlib.use('Agg')
import matplotlib.image as mpimg  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
from skimage import feature
from skimage import measure
from skimage import transform
from match_images_SH import match_images_delf

from tensorflow.python.platform import app
from delf import feature_io

cmd_args = None


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--image_1_path',
      type=str,
      default='test_images/image_1.jpg',
      help="""
      Path to test image 1.
      """)
  parser.add_argument(
      '--image_2_path',
      type=str,
      default='test_images/image_2.jpg',
      help="""
      Path to test image 2.
      """)
  parser.add_argument(
      '--features_1_path',
      type=str,
      default='test_features/image_1.delf',
      help="""
      Path to DELF features from image 1.
      """)
  parser.add_argument(
      '--features_2_path',
      type=str,
      default='test_features/image_2.delf',
      help="""
      Path to DELF features from image 2.
      """)
  parser.add_argument(
      '--output_image',
      type=str,
      default='test_match.png',
      help="""
      Path where an image showing the matches will be saved.
      """)
  cmd_args, unparsed = parser.parse_known_args()

  # need to loop through all pairs, get the inlier_count, then rank to produce the predictions output file
  # use compare_regions.py as inspiration

  inlier_count = match_images_delf(cmd_args.features_1_path, cmd_args.features_2_path)
