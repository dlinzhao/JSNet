# To estimate the mean instance size of each class in training set
import os
import sys
import numpy as np
from scipy import stats
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data', help='data dir')
parser.add_argument('--dataset', type=str, default='S3DIS', help='dataset [S3DIS]')
parser.add_argument('--input_list', type=str, default='data/train_hdf5_file_list_woArea5.txt', help='estimate the mean instance size')
parser.add_argument('--num_cls', type=int, default=13, help='estimate the mean instance size')
parser.add_argument('--out_dir', type=str, default='log5', help='log dir to save mean instance size [model path]')
FLAGS = parser.parse_args()


def estimate(flags):
    num_classes = flags.num_cls
    if flags.dataset == 'S3DIS':
        train_file_list = [os.path.join(flags.data_root, line.strip()) for line in open(flags.input_list, 'r')]
    else:
        print("Error: Not support the dataset: ", flags.dataset)
        return

    mean_ins_size = np.zeros(num_classes)
    ptsnum_in_gt = [[] for itmp in range(num_classes)]

    for h5_filename in train_file_list:
        print(h5_filename)
        cur_data, cur_group, _, cur_sem = provider.loadDataFile_with_groupseglabel_stanfordindoor(h5_filename)
        cur_data = np.reshape(cur_data, [-1, cur_data.shape[-1]])
        cur_group = np.reshape(cur_group, [-1])
        cur_sem = np.reshape(cur_sem, [-1])

        un = np.unique(cur_group)
        for ig, g in enumerate(un):
            tmp = (cur_group == g)
            sem_seg_g = int(stats.mode(cur_sem[tmp])[0])
            ptsnum_in_gt[sem_seg_g].append(np.sum(tmp))

    for idx in range(num_classes):
        mean_ins_size[idx] = np.mean(ptsnum_in_gt[idx]).astype(np.int)

    print(mean_ins_size)
    np.savetxt(os.path.join(flags.out_dir, 'mean_ins_size.txt'), mean_ins_size)


if __name__ == "__main__":
    estimate(FLAGS)
