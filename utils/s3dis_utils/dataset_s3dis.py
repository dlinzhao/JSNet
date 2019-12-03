import os
import sys
import time
import gc
import numpy as np

import multiprocessing
from concurrent import futures
from functools import partial as functools_partial

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import provider
import indoor3d_util


def data_sample(data_sample_queue, input_list, split, epoch, num_works, block_points=4096,
                block_size=1.0, stride=0.5, random_sample=False, sample_num=None, sample_aug=1):
    assert (input_list[0].endswith('npy') or input_list[0].endswith('h5')), "data format must be .npy or .h5"

    input_list_length = len(input_list)
    num_work = min(min(num_works, multiprocessing.cpu_count()), input_list_length // 4)

    if input_list_length > 4:
        num_work = max(num_work, 4)

    chunksize = input_list_length // num_work
    print("num input_list: {}, num works: {}, chunksize: {}".format(input_list_length, num_work, chunksize))

    if input_list[0].endswith('npy'):
        data_sample_func = functools_partial(
            indoor3d_util.room2blocks_wrapper_normalized, num_point=block_points, block_size=block_size,
            stride=stride, random_sample=random_sample, sample_num=sample_num, sample_aug=sample_aug)
    elif input_list[0].endswith('h5'):
        def load_data_file(input_file):
            cur_data, cur_group, _, cur_sem = provider.loadDataFile_with_groupseglabel_stanfordindoor(input_file)
            return cur_data, cur_sem, cur_group
        data_sample_func = load_data_file

    def data_sample_single(input_file):
        datalabel = data_sample_func(input_file)
        if split == 'train':
            datalabel = provider.shuffle_data(*datalabel)
        return datalabel

    for _ in range(epoch):
        np.random.shuffle(input_list)
        for idx in range(chunksize + 1):
            start_idx = min(idx * num_work, input_list_length)
            end_idx = min((idx + 1) * num_work, input_list_length)
            if start_idx >= input_list_length or end_idx > input_list_length:
                continue

            with futures.ThreadPoolExecutor(num_work) as pool:
                data_sem_ins = list(pool.map(data_sample_single, input_list[start_idx:end_idx], chunksize=1))

                for dsi in data_sem_ins:
                    shuffle_dsi = provider.shuffle_data(*dsi)
                    data_sample_queue.put(shuffle_dsi)
                    del dsi
                    gc.collect()

                pool.shutdown()
                gc.collect()


def data_prepare(data_sample_queue, data_queue, blocks, epoch, batch_size):
    data_list = list()
    sem_label_list = list()
    ins_label_list = list()

    total_batch = (blocks // batch_size) * epoch

    while total_batch > 0:
        data, sem_label, ins_label = data_sample_queue.get()

        data_list.append(data)
        sem_label_list.append(sem_label)
        ins_label_list.append(ins_label)

        del data
        del sem_label
        del ins_label

        batch_data = np.concatenate(data_list, axis=0)
        batch_sem_label = np.concatenate(sem_label_list, axis=0)
        batch_ins_label = np.concatenate(ins_label_list, axis=0)

        batch_data_length = batch_data.shape[0]
        num_batch_size = batch_data_length // batch_size
        for idx in range(num_batch_size):
            total_batch -= 1
            start_idx = idx * batch_size
            end_idx = (idx + 1) * batch_size
            data_queue.put((batch_data[start_idx: end_idx, ...],
                            batch_sem_label[start_idx: end_idx],
                            batch_ins_label[start_idx: end_idx]))

        remainder = batch_data_length % batch_size
        if remainder:
            data_list = [batch_data[-remainder:]]
            sem_label_list = [batch_sem_label[-remainder:]]
            ins_label_list = [batch_ins_label[-remainder:]]
        else:
            data_list = list()
            sem_label_list = list()
            ins_label_list = list()

        del batch_data
        del batch_sem_label
        del batch_ins_label

        gc.collect()


class S3DISDataset(object):
    def __init__(self, data_root, input_list_txt, split='train', epoch=1, batch_size=24, num_works=8,
                 data_type='numpy', block_points=4096, block_size=1.0, stride=0.5, random_sample=False,
                 sample_num=None, sample_aug=1, with_rgb=True):
        self.input_list_txt = input_list_txt
        self.split = split
        self.data_root = data_root
        self.data_type = data_type
        self.capacity = 30
        self.length = 0

        assert (data_type == 'numpy' or data_type == 'hdf5'), 'data_type must be "numpy" or "hdf5"'

        self.input_list = self.get_input_list()

        self.manager = multiprocessing.Manager()
        self.data_sample_queue = self.manager.Queue(3)
        self.data_queue = multiprocessing.Manager().Queue(self.capacity)

        self.producer_process = multiprocessing.Process(target=data_sample, args=(
            self.data_sample_queue, self.input_list, split, epoch, num_works,
            block_points, block_size, stride, random_sample, sample_num, sample_aug))

        self.consumer_process = multiprocessing.Process(target=data_prepare, args=(
            self.data_sample_queue, self.data_queue, self.length, epoch, batch_size))

        self.producer_process.start()
        self.consumer_process.start()

    def __del__(self):
        while not self.data_sample_queue.empty() and not self.data_queue.empty():
            self.data_queue.get_nowait()

        if self.producer_process.is_alive():
            self.producer_process.join()

        if self.consumer_process.is_alive():
            self.consumer_process.join()

    def __len__(self):
        return self.length

    def get_input_list(self):
        input_list = [line.strip() for line in open(self.input_list_txt, 'r')]
        temp_list = [item.split('/')[-1].strip('.h5').strip('.npy') for item in input_list]
        temp_input_list = [line.strip() for line in
                           open(os.path.join(self.data_root, 'data/indoor3d_ins_seg_hdf5/room_filelist.txt'), 'r')]
        cnt_length = 0
        for item in temp_input_list:
            if item in temp_list:
                cnt_length += 1

        del temp_input_list
        self.length = cnt_length
        input_list = [os.path.join(self.data_root, item) for item in input_list]

        return input_list

    def get_batch(self, data_aug=False):
        data, sem_label, ins_label = self.data_queue.get()

        if data_aug and self.split == 'train':
            data[:, :, 0:3] = provider.jitter_point_cloud(data[:, :, 0:3])

        return data, sem_label, ins_label

    def get_length(self):
        return self.__len__()


if __name__ == '__main__':
    batch_size = 24
    data_set = S3DISDataset(ROOT_DIR, 'data/test_file_list_Area1.txt', epoch=2)
    num_batch = data_set.get_length() // batch_size
    for epoch in range(2):
        for idx in range(num_batch):
            _, _, _ = data_set.get_batch()
            print('epoch/num_epoch: {}/{}; batch/num_batch: {}/{};'.format(epoch, 2, idx, num_batch))
            time.sleep(1)
    print('finish')
