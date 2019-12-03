import argparse
import os
import socket
import sys

import numpy as np
import tensorflow as tf
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from model import *
from test_utils import *
from log_util import get_logger
from clustering import cluster
import provider
import indoor3d_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--data_root', default='data', help='data dir [default: data]')
parser.add_argument('--input_list', type=str, default='data/test_hdf5_file_list_Area5.txt', help='Input data list file')
parser.add_argument('--data_type', type=str, default='hdf5', help='Test file type, hdf5 or numpy [default: hdf5]')
parser.add_argument('--model_path', type=str, default='log/model.ckpt', help='Path of model')
parser.add_argument('--log_dir', default='logs', help='Log dir [default: logs]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--bandwidth', type=float, default=1., help='Bandwidth for meanshift clustering [default: 1.]')
parser.add_argument('--verbose', action='store_true', help='if specified, output color-coded seg obj files')
FLAGS = parser.parse_args()

BATCH_SIZE = 1
FILE_TYPE = FLAGS.data_type
DATA_ROOT = FLAGS.data_root
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
MODEL_PATH = FLAGS.model_path
TEST_FILE_LIST = FLAGS.input_list
BANDWIDTH = FLAGS.bandwidth

mean_num_pts_in_group = np.loadtxt(os.path.join(os.path.dirname(MODEL_PATH), 'mean_ins_size.txt'))

output_verbose = FLAGS.verbose  # If true, output all color-coded segmentation obj files

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

OUTPUT_DIR = os.path.join(LOG_DIR, 'test_results')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

VIS_DIR = os.path.join(LOG_DIR, 'vis_results')
if FLAGS.verbose and not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

if os.path.exists('test.py'):
    os.system('cp test.py %s' % LOG_DIR)

logger = get_logger(__file__, LOG_DIR, 'log_inference.txt')
logger.info(str(FLAGS) + '\n')

MAX_NUM_POINT = 4096
NUM_CLASSES = 13
NEW_NUM_CLASSES = 13

HOSTNAME = socket.gethostname()

EXT_LEN = 4
if FILE_TYPE == 'hdf5':
    EXT_LEN = 3
elif FILE_TYPE == 'numpy':
    EXT_LEN = 4
else:
    raise Exception('Not support file type')

ROOM_PATH_LIST = [os.path.join(ROOT_DIR, line.rstrip()) for line in open(os.path.join(ROOT_DIR, FLAGS.input_list))]
len_pts_files = len(ROOM_PATH_LIST)


def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl, sem_labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Get model
            pred_sem, pred_ins = get_model(pointclouds_pl, is_training_pl, NUM_CLASSES)
            pred_sem_softmax = tf.nn.softmax(pred_sem)
            pred_sem_label = tf.argmax(pred_sem_softmax, axis=2)

            loader = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        is_training = False

        # Restore variables from disk.
        loader.restore(sess, MODEL_PATH)
        logger.info("Model restored from {}".format(MODEL_PATH))

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'sem_labels_pl': sem_labels_pl,
               'is_training_pl': is_training_pl,
               'pred_ins': pred_ins,
               'pred_sem_label': pred_sem_label,
               'pred_sem_softmax': pred_sem_softmax}

        total_acc = 0.0
        total_seen = 0

        output_filelist_f = os.path.join(LOG_DIR, 'output_filelist.txt')
        fout_out_filelist = []
        for shape_idx in range(len_pts_files):
            room_path = ROOM_PATH_LIST[shape_idx]

            out_data_label_filename = os.path.basename(room_path)[:-EXT_LEN] + '_pred.txt'
            out_data_label_filename = os.path.join(OUTPUT_DIR, out_data_label_filename)
            out_gt_label_filename = os.path.basename(room_path)[:-EXT_LEN] + '_gt.txt'
            out_gt_label_filename = os.path.join(OUTPUT_DIR, out_gt_label_filename)
            fout_data_label = []
            fout_gt_label = []

            fout_out_filelist.append(out_data_label_filename + '\n')

            logger.info('%d / %d ...' % (shape_idx, len_pts_files))
            logger.info('Loading file ' + room_path)

            size_path = room_path
            if FILE_TYPE == 'hdf5':
                size_path = size_path.replace('indoor3d_ins_seg_hdf5', 'stanford_indoor3d_ins.sem')
                size_path = "{}.npy".format(size_path[:-3])
                cur_data, cur_group, _, cur_sem = \
                    provider.loadDataFile_with_groupseglabel_stanfordindoor(room_path)
            elif FILE_TYPE == 'numpy':
                cur_data, cur_sem, cur_group = \
                    indoor3d_util.room2blocks_wrapper_normalized(room_path, NUM_POINT, block_size=1.0, stride=0.5,
                                                                 random_sample=False, sample_num=None)
            cur_data = cur_data[:, 0:NUM_POINT, :]
            cur_sem = np.squeeze(cur_sem)
            cur_group = np.squeeze(cur_group)
            # Get room dimension..
            data_label = np.load(size_path)
            data = data_label[:, 0:6]
            max_room_x = max(data[:, 0])
            max_room_y = max(data[:, 1])
            max_room_z = max(data[:, 2])

            cur_pred_sem = np.zeros_like(cur_sem)
            cur_pred_sem_softmax = np.zeros([cur_sem.shape[0], cur_sem.shape[1], NUM_CLASSES])
            group_output = np.zeros_like(cur_group)

            gap = 5e-3
            volume_num = int(1. / gap) + 1
            volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
            volume_seg = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)

            num_data = cur_data.shape[0]
            for j in range(num_data):
                logger.info("Processsing: Shape [%d] Block[%d]" % (shape_idx, j))

                pts = cur_data[j, ...]
                group = cur_group[j]
                sem = cur_sem[j]

                feed_dict = {ops['pointclouds_pl']: np.expand_dims(pts, 0),
                             ops['labels_pl']: np.expand_dims(group, 0),
                             ops['sem_labels_pl']: np.expand_dims(sem, 0),
                             ops['is_training_pl']: is_training}

                pred_ins_val, pred_sem_label_val, pred_sem_softmax_val = sess.run(
                    [ops['pred_ins'], ops['pred_sem_label'], ops['pred_sem_softmax']], feed_dict=feed_dict)

                pred_val = np.squeeze(pred_ins_val, axis=0)
                pred_sem = np.squeeze(pred_sem_label_val, axis=0)
                pred_sem_softmax = np.squeeze(pred_sem_softmax_val, axis=0)
                cur_pred_sem[j, :] = pred_sem
                cur_pred_sem_softmax[j, ...] = pred_sem_softmax

                # cluster
                group_seg = {}
                bandwidth = BANDWIDTH
                num_clusters, labels, cluster_centers = cluster(pred_val, bandwidth)
                for idx_cluster in range(num_clusters):
                    tmp = (labels == idx_cluster)
                    estimated_seg = int(stats.mode(pred_sem[tmp])[0])
                    group_seg[idx_cluster] = estimated_seg

                groupids_block = labels

                groupids = BlockMerging(volume, volume_seg, pts[:, 6:],
                                        groupids_block.astype(np.int32), group_seg, gap)

                group_output[j, :] = groupids
                total_acc += float(np.sum(pred_sem == sem)) / pred_sem.shape[0]
                total_seen += 1

            group_pred = group_output.reshape(-1)
            seg_pred = cur_pred_sem.reshape(-1)
            seg_pred_softmax = cur_pred_sem_softmax.reshape([-1, NUM_CLASSES])
            pts = cur_data.reshape([-1, 9])

            # filtering
            x = (pts[:, 6] / gap).astype(np.int32)
            y = (pts[:, 7] / gap).astype(np.int32)
            z = (pts[:, 8] / gap).astype(np.int32)
            for i in range(group_pred.shape[0]):
                if volume[x[i], y[i], z[i]] != -1:
                    group_pred[i] = volume[x[i], y[i], z[i]]

            seg_gt = cur_sem.reshape(-1)
            un = np.unique(group_pred)
            pts_in_pred = [[] for itmp in range(NUM_CLASSES)]
            group_pred_final = -1 * np.ones_like(group_pred)
            grouppred_cnt = 0
            for ig, g in enumerate(un):  # each object in prediction
                if g == -1:
                    continue
                tmp = (group_pred == g)
                sem_seg_g = int(stats.mode(seg_pred[tmp])[0])
                # if np.sum(tmp) > 500:
                if np.sum(tmp) > 0.25 * mean_num_pts_in_group[sem_seg_g]:
                    group_pred_final[tmp] = grouppred_cnt
                    pts_in_pred[sem_seg_g] += [tmp]
                    grouppred_cnt += 1

            pts[:, 6] *= max_room_x
            pts[:, 7] *= max_room_y
            pts[:, 8] *= max_room_z
            pts[:, 3:6] *= 255.0
            ins = group_pred_final.astype(np.int32)
            sem = seg_pred.astype(np.int32)
            sem_softmax = seg_pred_softmax
            sem_gt = seg_gt
            ins_gt = cur_group.reshape(-1)

            for i in range(pts.shape[0]):
                fout_data_label.append('%f %f %f %d %d %d %f %d %d\n' % (
                    pts[i, 6], pts[i, 7], pts[i, 8], pts[i, 3], pts[i, 4], pts[i, 5], sem_softmax[i, sem[i]],
                    sem[i], ins[i]))
                fout_gt_label.append('%d %d\n' % (sem_gt[i], ins_gt[i]))

            with open(out_data_label_filename, 'w') as fd:
                fd.writelines(fout_data_label)
            with open(out_gt_label_filename, 'w') as fd:
                fd.writelines(fout_gt_label)

            if output_verbose:
                # file name
                outfile_name = ROOM_PATH_LIST[shape_idx].split('/')[-1][:-EXT_LEN]
                # Raw Point Cloud
                output_point_cloud_rgb(pts[:, 6:], pts[:, 3:6].astype(np.int32), os.path.join(VIS_DIR, '{}_raw.obj'.format(outfile_name)))
                logger.info('Saving file {}_raw.obj'.format(outfile_name))
                # Instance Prediction
                output_color_point_cloud(pts[:, 6:], group_pred_final.astype(np.int32), os.path.join(VIS_DIR, '{}_pred_ins.obj'.format(outfile_name)))
                logger.info('Saving file {}_pred_ins.obj'.format(outfile_name))
                # Semantic Prediction
                output_color_point_cloud(pts[:, 6:], seg_pred.astype(np.int32), os.path.join(VIS_DIR, '{}_pred_sem.obj'.format(outfile_name)))
                logger.info('Saving file {}_pred_sem.obj'.format(outfile_name))
                # Instance Ground Truth
                output_color_point_cloud(pts[:, 6:], ins_gt, os.path.join(VIS_DIR, '{}_gt_ins.obj'.format(outfile_name)))
                logger.info('Saving file {}_gt_ins.obj'.format(outfile_name))
                # Semantic Ground Truth
                output_color_point_cloud(pts[:, 6:], sem_gt, os.path.join(VIS_DIR, '{}_gt_sem.obj'.format(outfile_name)))
                logger.info('Saving file {}_gt_sem.obj'.format(outfile_name))

        with open(output_filelist_f, 'w') as fd:
            fd.writelines(fout_out_filelist)


if __name__ == "__main__":
    test()
