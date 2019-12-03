import os
import sys

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module, pointnet_upsample
from pointconv_util import pointconv_encoding, pointconv_decoding_depthwise
from loss import *


def placeholder_inputs(batch_size, num_point, num_dims=9):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_dims))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    sem_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, sem_pl


def get_model(point_cloud, is_training, num_class, num_embed=5, sigma=0.05, bn_decay=None, is_dist=False):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud[:, :, :3]
    l0_points = point_cloud[:, :, 3:]
    end_points['l0_xyz'] = l0_xyz

    # shared encoder
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=1024, radius=0.1, nsample=32, mlp=[32, 32, 64], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, scope='layer1')
    l2_xyz, l2_points = pointconv_encoding(l1_xyz, l1_points, npoint=256, radius=0.2, sigma=2 * sigma, K=32, mlp=[ 64,  64, 128], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='layer2')
    l3_xyz, l3_points = pointconv_encoding(l2_xyz, l2_points, npoint=64,  radius=0.4, sigma=4 * sigma, K=32, mlp=[128, 128, 256], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='layer3')
    l4_xyz, l4_points = pointconv_encoding(l3_xyz, l3_points, npoint=32,  radius=0.8, sigma=8 * sigma, K=32, mlp=[256, 256, 512], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='layer4')

    # semantic decoder
    l3_points_sem = pointconv_decoding_depthwise(l3_xyz, l4_xyz, l3_points, l4_points,     radius=0.8, sigma=8*sigma, K=16, mlp=[512, 512], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='sem_fa_layer1')
    l2_points_sem = pointconv_decoding_depthwise(l2_xyz, l3_xyz, l2_points, l3_points_sem, radius=0.4, sigma=4*sigma, K=16, mlp=[256, 256], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='sem_fa_layer2')  # 48x256x256
    l1_points_sem = pointconv_decoding_depthwise(l1_xyz, l2_xyz, l1_points, l2_points_sem, radius=0.2, sigma=2*sigma, K=16, mlp=[256, 128], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='sem_fa_layer3')  # 48x1024x128
    l0_points_sem = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_sem, [128, 128, 128], is_training, bn_decay, is_dist=is_dist, scope='sem_fa_layer4')  # 48x4096x128

    # instance decoder
    l3_points_ins = pointconv_decoding_depthwise(l3_xyz, l4_xyz, l3_points, l4_points,     radius=0.8, sigma=8*sigma, K=16, mlp=[512, 512], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='ins_fa_layer1')
    l2_points_ins = pointconv_decoding_depthwise(l2_xyz, l3_xyz, l2_points, l3_points_ins, radius=0.4, sigma=4*sigma, K=16, mlp=[256, 256], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='ins_fa_layer2')  # 48x256x256
    l1_points_ins = pointconv_decoding_depthwise(l1_xyz, l2_xyz, l1_points, l2_points_ins, radius=0.2, sigma=2*sigma, K=16, mlp=[256, 128], is_training=is_training, bn_decay=bn_decay, is_dist=is_dist, weight_decay=None, scope='ins_fa_layer3')  # 48x1024x128
    l0_points_ins = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points_ins, [128, 128, 128], is_training, bn_decay, is_dist=is_dist, scope='ins_fa_layer4')   # 48x4096x128

    # FC layers F_sem
    l2_points_sem_up = pointnet_upsample(l0_xyz, l2_xyz, l2_points_sem, scope='sem_up1')
    l1_points_sem_up = pointnet_upsample(l0_xyz, l1_xyz, l1_points_sem, scope='sem_up2')
    net_sem_0 = tf.add(tf.concat([l0_points_sem, l1_points_sem_up], axis=-1, name='sem_up_concat'), l2_points_sem_up, name='sem_up_add')
    net_sem_0 = tf_util.conv1d(net_sem_0, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_fc1', bn_decay=bn_decay)

    # FC layers F_ins
    l2_points_ins_up = pointnet_upsample(l0_xyz, l2_xyz, l2_points_ins, scope='ins_up1')
    l1_points_ins_up = pointnet_upsample(l0_xyz, l1_xyz, l1_points_ins, scope='ins_up2')
    net_ins_0 = tf.add(tf.concat([l0_points_ins, l1_points_ins_up], axis=-1, name='ins_up_concat'), l2_points_ins_up, name='ins_up_add')
    net_ins_0 = tf_util.conv1d(net_ins_0, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fc1', bn_decay=bn_decay)

    # Adaptation
    net_sem_cache_0 = tf_util.conv1d(net_sem_0, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_cache_1', bn_decay=bn_decay)
    net_ins_1 = net_ins_0 + net_sem_cache_0

    net_ins_2 = tf.concat([net_ins_0, net_ins_1], axis=-1, name='net_ins_2_concat')
    net_ins_atten = tf.sigmoid(tf.reduce_mean(net_ins_2, axis=-1, keep_dims=True, name='ins_reduce'), name='ins_atten')
    net_ins_3 = net_ins_2 * net_ins_atten

    # Aggregation
    net_ins_cache_0 = tf_util.conv1d(net_ins_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_cache_1', bn_decay=bn_decay)
    net_ins_cache_1 = tf.reduce_mean(net_ins_cache_0, axis=1, keep_dims=True, name='ins_cache_2')
    net_ins_cache_1 = tf.tile(net_ins_cache_1, [1, num_point, 1], name='ins_cache_tile')
    net_sem_1 = net_sem_0 + net_ins_cache_1

    net_sem_2 = tf.concat([net_sem_0, net_sem_1], axis=-1, name='net_sem_2_concat')
    net_sem_atten = tf.sigmoid(tf.reduce_mean(net_sem_2, axis=-1, keep_dims=True, name='sem_reduce'), name='sem_atten')
    net_sem_3 = net_sem_2 * net_sem_atten

    # Output
    net_ins_3 = tf_util.conv1d(net_ins_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='ins_fc2', bn_decay=bn_decay)
    net_ins_4 = tf_util.dropout(net_ins_3, keep_prob=0.5, is_training=is_training, scope='ins_dp_4')
    net_ins_4 = tf_util.conv1d(net_ins_4, num_embed, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='ins_fc5')

    net_sem_3 = tf_util.conv1d(net_sem_3, 128, 1, padding='VALID', bn=True, is_training=is_training, is_dist=is_dist, scope='sem_fc2', bn_decay=bn_decay)
    net_sem_4 = tf_util.dropout(net_sem_3, keep_prob=0.5, is_training=is_training, scope='sem_dp_4')
    net_sem_4 = tf_util.conv1d(net_sem_4, num_class, 1, padding='VALID', activation_fn=None, is_dist=is_dist, scope='sem_fc5')

    return net_sem_4, net_ins_4


def get_loss(pred, ins_label, pred_sem_label, pred_sem, sem_label, weights=1.0, disc_weight=1.0):
    """ pred:   BxNxE,
        ins_label:  BxN
        pred_sem_label: BxN
        pred_sem: BxNx13
        sem_label: BxN
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=sem_label, logits=pred_sem, weights=weights)
    tf.summary.scalar('classify loss', classify_loss)

    feature_dim = pred.get_shape()[-1]
    delta_v = 0.5
    delta_d = 1.5
    param_var = 1.
    param_dist = 1.

    disc_loss, l_var, l_dist = discriminative_loss(pred, ins_label, feature_dim, delta_v, delta_d, param_var, param_dist)

    disc_loss = disc_loss * disc_weight
    l_var = l_var * disc_weight
    l_dist = l_dist * disc_weight
    
    loss = classify_loss + disc_loss

    tf.add_to_collection('losses', loss)
    return loss, classify_loss, disc_loss, l_var, l_dist


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 2048, 3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
