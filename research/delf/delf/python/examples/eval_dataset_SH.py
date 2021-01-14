
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
import os

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
from collections import namedtuple
from scipy.io import loadmat, savemat

from tensorflow.python.platform import app
from delf import feature_io
from tqdm.auto import tqdm

cmd_args = None


dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr',
                                   'dbTimeStamp', 'qTimeStamp', 'gpsDb', 'gpsQ'])

def parse_db_struct(path):
    mat = loadmat(path)

    fieldnames = list(mat['dbStruct'][0, 0].dtype.names)

    if 'dataset' in fieldnames:
        dataset = mat['dbStruct'][0, 0]['dataset'].item()
    else:
        if '250k' in path.split('/')[-1].lower():
            dataset = 'pitts250k'
        elif '30k' in path.split('/')[-1].lower():
            dataset = 'pitts30k'
        elif 'tokyoTM' in path.split('/')[-1].lower():
            dataset = 'tokyoTM'
        elif 'oxford' in path.split('/')[-1].lower():
            dataset = 'oxford'
        elif 'kudamm' in path.split('/')[-1].lower():
            dataset = 'kudamm'
        elif 'nordland' in path.split('/')[-1].lower():
            dataset = 'nordland'
        else:
            raise ValueError('Dataset not supported')

    whichSet = mat['dbStruct'][0, 0]['whichSet'].item()

    dbImage = [f[0].item() for f in mat['dbStruct'][0, 0]['dbImageFns']]

    qImage = [f[0].item() for f in mat['dbStruct'][0, 0]['qImageFns']]

    if dataset == 'tokyo247':
        dbImage = [im.replace('.jpg', '.png') for im in dbImage]

    numDb = mat['dbStruct'][0, 0]['numImages'].item()
    numQ = mat['dbStruct'][0, 0]['numQueries'].item()

    posDistThr = mat['dbStruct'][0, 0]['posDistThr'].item()
    posDistSqThr = mat['dbStruct'][0, 0]['posDistSqThr'].item()
    if 'nonTrivPosDistSqThr' in fieldnames:
        nonTrivPosDistSqThr = mat['dbStruct'][0, 0]['nonTrivPosDistSqThr'].item()
    else:
        nonTrivPosDistSqThr = None

    if 'dbTimeStamp' in fieldnames and 'qTimeStamp' in fieldnames:
        dbTimeStamp = [f[0].item() for f in mat['dbStruct'][0, 0]['dbTimeStamp'].T]
        qTimeStamp = [f[0].item() for f in mat['dbStruct'][0, 0]['qTimeStamp'].T]
        dbTimeStamp = np.array(dbTimeStamp)
        qTimeStamp = np.array(qTimeStamp)
    else:
        dbTimeStamp = None
        qTimeStamp = None

    if 'utmQ' in fieldnames and 'utmDb' in fieldnames:
        utmDb = mat['dbStruct'][0, 0]['utmDb'].T
        utmQ = mat['dbStruct'][0, 0]['utmQ'].T
    else:
        utmQ = None
        utmDb = None

    if 'gpsQ' in fieldnames and 'gpsDb' in fieldnames:
        gpsDb = mat['dbStruct'][0, 0]['gpsDb'].T
        gpsQ = mat['dbStruct'][0, 0]['gpsQ'].T
    else:
        gpsQ = None
        gpsDb = None

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage, utmQ, numDb, numQ, posDistThr,
                    posDistSqThr, nonTrivPosDistSqThr, dbTimeStamp, qTimeStamp, gpsQ, gpsDb)


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
        '--features_db_path',
        type=str,
        default='/work/qvpr/workspace/delf/Nordland/Database',
        help="""
      Path to folder full of DELF features from database.
      """)
    parser.add_argument(
        '--features_q_path',
        type=str,
        default='/work/qvpr/workspace/delf/Nordland/Query',
        help="""
      Path to folder full of DELF features from query.
      """)
    parser.add_argument(
        '--output_image',
        type=str,
        default='test_match.png',
        help="""
      Path where an image showing the matches will be saved.
      """)
    parser.add_argument(
        '--predictions_input',
        type=str,
        default='predictions.txt',
        help="""
      Path to predictions input file.
      """)
    parser.add_argument(
        '--out_save_path',
        type=str,
        default='predictions.txt',
        help="""
      Path to save output files to.
      """)
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='nordland', #valid options: kudamm, pittsburgh, tokyo247, nordland, mapillarysf, mapillarycph
        help="""
      Name of dataset being evaluated, used to prefix output files.
      """)
    parser.add_argument(
        '--path_to_mat_struct_file',
        type=str,
        default='kudamm.mat',
        help="""
          Full path to dataset struct mat file.
          """)

    cmd_args, unparsed = parser.parse_known_args()

    # need to loop through all pairs, get the inlier_count, then rank to produce the predictions output file
    # use compare_regions.py as inspiration

    dbstruct = parse_db_struct(cmd_args.path_to_mat_struct_file)

    mydatasetname = cmd_args.dataset_name

    ref_root_to_remove = ''
    query_root_to_remove = ''

    if mydatasetname in ['mapillarysf', 'mapillarycph']:
        ref_root_to_remove = 'train_val/' + mydatasetname.replace('mapillary', '') + '/'
        query_root_to_remove = 'train_val/' + mydatasetname.replace('mapillary', '') + '/'

    if mydatasetname == 'kudamm':
        qImage = ['Query/' + qim.replace(query_root_to_remove, '') for qim in dbstruct.qImage]
        dbImage = ['Reference/' + rim.replace(ref_root_to_remove, '') for rim in dbstruct.dbImage]
    elif mydatasetname == 'pittsburgh':
        qImage = ['queries_real/' + qim.replace(query_root_to_remove, '') for qim in dbstruct.qImage]
        dbImage = [rim.replace(ref_root_to_remove, '') for rim in dbstruct.dbImage]
    elif mydatasetname == 'tokyo247':
        qImage = ['247query_v3/' + qim.replace(query_root_to_remove, '') for qim in dbstruct.qImage]
        dbImage = [rim.replace(ref_root_to_remove, '') for rim in dbstruct.dbImage]
    elif mydatasetname == 'nordland':
        qImage = ['winter/' + qim.replace(query_root_to_remove, '') for qim in dbstruct.qImage]
        dbImage = ['summer/' + rim.replace(ref_root_to_remove, '') for rim in dbstruct.dbImage]
    else:
        qImage = [qim.replace(query_root_to_remove, '') for qim in dbstruct.qImage]
        dbImage = [rim.replace(ref_root_to_remove, '') for rim in dbstruct.dbImage]

    output_file = cmd_args.out_save_path + '/' + cmd_args.dataset_name + '_delf_predictions.npy'

    skip_rows = 2

    with open(cmd_args.predictions_input, 'r') as f:
        for _ in range(skip_rows):
            f.readline()
        pairs = [l.split() for l in f.readlines()]

    predictions = {}

    assert '.npy' in output_file
    output_prediction_filepath = output_file.replace('.npy', '_match_pairs.txt')
    with open(output_prediction_filepath, 'w') as outfile:
        outfile.write('# kapture format: 1.0\n')
        outfile.write('# query_image, map_image, score\n')
        for pair in tqdm(pairs):
            name0, name1 = pair[:2]
            if name0.endswith(','):
                name0 = name0[:-1]
            if name1.endswith(','):
                name1 = name1[:-1]

            name0short = os.path.basename(name0) #this may need changing depending on how the delf features are saved
            name1short = os.path.basename(name1)

            #relying on assumption that all kapture files go query then database

            file_q = cmd_args.features_q_path + '/' + name0short[:-4] + '.delf'
            file_db = cmd_args.features_db_path + '/' + name1short[:-4] + '.delf'

            inlier_count = match_images_delf(file_q, file_db)
            inlier_count = 0

            if name0 not in predictions:
                predictions[name0] = {}

            predictions[name0][name1] = inlier_count
            outfile.write(name0 + ', ' + name1 + ', ' + str(inlier_count) + '\n')

    pred_out = []
    for qidx, qim in enumerate(predictions):
        pred_query = []
        for ridx in np.argsort(np.array(list(predictions[qim].values())))[::-1]:
            pred_query.append(dbImage.index(list(predictions[qim].keys())[ridx]))
        pred_out.append(pred_query)
    pred_out = np.array(pred_out)

    np.save(output_file, pred_out)












