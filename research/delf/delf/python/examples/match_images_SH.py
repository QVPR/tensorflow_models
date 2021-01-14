# Lint as: python3
# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Matches two images using their DELF features.

The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.

The DELF features can be extracted using the extract_features.py script.
"""

# SH edits
# This new file replaces the python main.py with a function call instead.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

sys.path.append('../')

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

from tensorflow.python.platform import app
from delf import feature_io #need to install delf locally on terminal...

cmd_args = None

_DISTANCE_THRESHOLD = 0.8


def match_images_delf(features_1_path, features_2_path):
  # Read features.
  locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(features_1_path)
  num_features_1 = locations_1.shape[0]
  print(f"Loaded image 1's {num_features_1} features")
  locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(features_2_path)
  num_features_2 = locations_2.shape[0]
  print(f"Loaded image 2's {num_features_2} features")

  # Find nearest-neighbor matches using a KD tree.
  d1_tree = spatial.cKDTree(descriptors_1)
  _, indices = d1_tree.query(
      descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

  # Select feature locations for putative matches.
  locations_2_to_use = np.array([
      locations_2[i,]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])
  locations_1_to_use = np.array([
      locations_1[indices[i],]
      for i in range(num_features_2)
      if indices[i] != num_features_1
  ])

  # Perform geometric verification using RANSAC.
  _, inliers = measure.ransac((locations_1_to_use, locations_2_to_use),
                              transform.AffineTransform,
                              min_samples=3,
                              residual_threshold=20,
                              max_trials=1000)

  inlier_count = sum(inliers)
  #print(f'Found {inlier_count} inliers')

  return inlier_count


