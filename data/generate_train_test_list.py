import os
import glob

input_dir = "stanford_indoor3d_ins.sem/"

for i in range(6):

    filter_id = i + 1

    # For training
    train_fold_pattern = "Area_[!{}]*.npy".format(filter_id)
    dst_list_path = "train_file_list_woArea{}.txt".format(filter_id)

    npy_files = glob.glob(os.path.join(input_dir + train_fold_pattern))

    f = open(dst_list_path, 'w')
    for npy_file in npy_files:
        f.write('data/' + npy_file)
        f.write('\n')
    f.close()

    # For test
    test_fold_pattern = "Area_{}*.npy".format(filter_id)
    dst_list_path = "test_file_list_Area{}.txt".format(filter_id)

    npy_files = glob.glob(os.path.join(input_dir + test_fold_pattern))

    f = open(dst_list_path, 'w')
    for npy_file in npy_files:
        f.write('data/' + npy_file)
        f.write('\n')
    f.close()
