"""
Helper Function for PointConv
Author: Wenxuan Wu
Date: July 2018
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))

from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point
from tf_interpolate import three_nn, three_interpolate
import tf_util


def knn_kdtree(nsample, xyz, new_xyz):
    batch_size = xyz.shape[0]
    n_points = new_xyz.shape[1]

    indices = np.zeros((batch_size, n_points, nsample), dtype=np.int32)
    for batch_idx in range(batch_size):
        X = xyz[batch_idx, ...]
        q_X = new_xyz[batch_idx, ...]
        kdt = KDTree(X, leaf_size=30)
        _, indices[batch_idx] = kdt.query(q_X, k=nsample)

    return indices


def kernel_density_estimation_ball(pts, radius, sigma, N_points=128, is_norm=False):
    with tf.variable_scope("ComputeDensity") as sc:
        idx, pts_cnt = query_ball_point(radius, N_points, pts, pts)
        g_pts = group_point(pts, idx)
        g_pts -= tf.tile(tf.expand_dims(pts, 2), [1, 1, N_points, 1])

        R = tf.sqrt(sigma)
        xRinv = tf.div(g_pts, R)
        quadform = tf.reduce_sum(tf.square(xRinv), axis=-1)
        logsqrtdetSigma = tf.log(R) * 3
        mvnpdf = tf.exp(-0.5 * quadform - logsqrtdetSigma - 3 * tf.log(2 * 3.1415926) / 2)

        first_val, _ = tf.split(mvnpdf, [1, N_points - 1], axis=2)

        mvnpdf = tf.reduce_sum(mvnpdf, axis=2, keep_dims=True)

        num_val_to_sub = tf.expand_dims(tf.cast(tf.subtract(N_points, pts_cnt), dtype=tf.float32), axis=-1)

        val_to_sub = tf.multiply(first_val, num_val_to_sub)

        mvnpdf = tf.subtract(mvnpdf, val_to_sub)

        scale = tf.div(1.0, tf.expand_dims(tf.cast(pts_cnt, dtype=tf.float32), axis=-1))
        density = tf.multiply(mvnpdf, scale)

        if is_norm:
            # grouped_xyz_sum = tf.reduce_sum(grouped_xyz, axis = 1, keepdims = True)
            density_max = tf.reduce_max(density, axis=1, keep_dims=True)
            density = tf.div(density, density_max)

        return density


def pointconv_sampling(npoint, pts):
    """
    inputs:
    npoint: scalar, number of points to sample
    pointcloud: B * N * 3, input point cloud
    output:
    sub_pts: B * npoint * 3, sub-sampled point cloud
    """

    sub_pts = gather_point(pts, farthest_point_sample(npoint, pts))
    return sub_pts


def pointconv_grouping(feature, K, src_xyz, q_xyz, use_xyz=True):
    """
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    """

    batch_size = src_xyz.get_shape()[0]
    npoint = q_xyz.get_shape()[1]

    point_indices = tf.py_func(knn_kdtree, [K, src_xyz, q_xyz], tf.int32)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
    idx = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
    idx.set_shape([batch_size, npoint, K, 2])

    grouped_xyz = tf.gather_nd(src_xyz, idx)
    grouped_xyz -= tf.tile(tf.expand_dims(q_xyz, 2), [1, 1, K, 1])  # translation normalization

    grouped_feature = tf.gather_nd(feature, idx)
    if use_xyz:
        new_points = tf.concat([grouped_xyz, grouped_feature], axis=-1)
    else:
        new_points = grouped_feature

    return grouped_xyz, new_points, idx


def weight_net_hidden(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay=None,
                      activation_fn=tf.nn.relu, is_dist=False):
    with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=True, is_training=is_training, activation_fn=activation_fn,
                                 scope='wconv{}'.format(i), bn_decay=bn_decay,
                                 weight_decay=weight_decay, is_dist=is_dist)

            # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='wconv_dp{}'.format(i))
    return net


def weight_net(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay=None,
               activation_fn=tf.nn.relu, is_dist=False):
    with tf.variable_scope(scope) as sc:
        net = xyz
        for i, num_hidden_units in enumerate(hidden_units):
            if i != len(hidden_units) - 1:
                net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=True, is_training=is_training, activation_fn=activation_fn,
                                     scope='wconv{}'.format(i), bn_decay=bn_decay,
                                     weight_decay=weight_decay, is_dist=is_dist)
            else:
                net = tf_util.conv2d(net, num_hidden_units, [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=False, is_training=is_training, activation_fn=None,
                                     scope='wconv{}'.format(i), bn_decay=bn_decay,
                                     weight_decay=weight_decay, is_dist=is_dist)
            # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='wconv_dp{}'.format(i))
    return net


def nonlinear_transform(data_in, mlp, scope, is_training, bn_decay=None, weight_decay=None,
                        activation_fn=tf.nn.relu, is_dist=False):
    with tf.variable_scope(scope) as sc:

        net = data_in
        l = len(mlp)
        if l > 1:
            for i, out_ch in enumerate(mlp[0:(l - 1)]):
                net = tf_util.conv2d(net, out_ch, [1, 1],
                                     padding='VALID', stride=[1, 1],
                                     bn=True, is_training=is_training, activation_fn=tf.nn.relu,
                                     scope='nonlinear{}'.format(i), bn_decay=bn_decay,
                                     weight_decay=weight_decay, is_dist=is_dist)

                # net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp_nonlinear{}'.format(i))
        net = tf_util.conv2d(net, mlp[-1], [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=False, is_training=is_training,
                             scope='nonlinear%d' % (l - 1), bn_decay=bn_decay,
                             activation_fn=tf.nn.sigmoid, weight_decay=weight_decay, is_dist=is_dist)

    return net


def pointconv_encoding(xyz, feature, npoint, radius, sigma, K, mlp, is_training, bn_decay, weight_decay, scope,
                       bn=True, use_xyz=True, is_dist=False):
    """ Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    """
    with tf.variable_scope(scope) as sc:
        num_points = xyz.get_shape()[1]
        if num_points == npoint:
            new_xyz = xyz
        else:
            new_xyz = pointconv_sampling(npoint, xyz)
        # grouped_feature: B x N x K x C
        grouped_xyz, grouped_feature, idx = pointconv_grouping(feature, K, xyz, new_xyz)

        density = kernel_density_estimation_ball(xyz, radius, sigma)
        inverse_density = tf.div(1.0, density)
        grouped_density = tf.gather_nd(inverse_density, idx)  # (batch_size, npoint, nsample, 1)
        # grouped_density = tf_grouping.group_point(inverse_density, idx)
        inverse_max_density = tf.reduce_max(grouped_density, axis=2, keep_dims=True)
        density_scale = tf.div(grouped_density, inverse_max_density)

        # density_scale = tf_grouping.group_point(density, idx)

        for i, num_out_channel in enumerate(mlp):
            if i != len(mlp) - 1:
                grouped_feature = tf_util.conv2d(grouped_feature, num_out_channel, [1, 1],
                                                 padding='VALID', stride=[1, 1],
                                                 bn=bn, is_training=is_training,
                                                 scope='conv{}'.format(i), is_dist=is_dist,
                                                 bn_decay=bn_decay, weight_decay=weight_decay)

        weight = weight_net_hidden(grouped_xyz, [32], scope='weight_net', is_training=is_training, bn_decay=bn_decay,
                                   weight_decay=weight_decay, is_dist=is_dist)

        density_scale = nonlinear_transform(density_scale, [16, 1], scope='density_net', is_training=is_training,
                                            bn_decay=bn_decay, weight_decay=weight_decay, is_dist=is_dist)

        # grouped_feature: B x N x K x C, density_scale: B x N x K x 1, new_points: B x N x K x C
        new_points = tf.multiply(grouped_feature, density_scale)
        # new_points: B x N x C x K
        new_points = tf.transpose(new_points, [0, 1, 3, 2])
        # weight: B x N x K x K
        # new_points: B x N x C x K
        new_points = tf.matmul(new_points, weight)
        # new_points = tf.transpose(new_points, [0, 1, 3, 2])

        new_points = tf_util.conv2d(new_points, mlp[-1], [1, new_points.get_shape()[2].value],
                                    padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                    scope='after_conv', bn_decay=bn_decay, is_dist=is_dist,
                                    weight_decay=weight_decay)

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])

        return new_xyz, new_points


def pointconv_decoding_depthwise(xyz1, xyz2, points1, points2, radius, sigma, K, mlp, is_training, bn_decay,
                                 weight_decay, scope, bn=True, use_xyz=True, is_dist=False):
    """ Input:
            depthwise version of pointconv
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            sigma: float32 -- KDE bandwidth
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    """
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        # setup for deConv
        grouped_xyz, grouped_feature, idx = pointconv_grouping(interpolated_points, K, xyz1, xyz1, use_xyz=use_xyz)

        weight = weight_net(grouped_xyz, [32, grouped_feature.get_shape()[3].value], scope='decode_weight_net',
                            is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay, is_dist=is_dist)

        new_points = tf.multiply(grouped_feature, weight)

        new_points = tf_util.reduce_sum2d_conv(new_points, axis=2, scope='fp_sumpool', bn=True, is_dist=is_dist,
                                               bn_decay=bn_decay, is_training=is_training, keepdims=False)

        if points1 is not None:
            # [B x ndataset1 x nchannel1+nchannel2]
            new_points1 = tf.concat(axis=-1, values=[new_points, points1])
        else:
            new_points1 = new_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training, is_dist=is_dist,
                                         scope='conv_%d' % (i), bn_decay=bn_decay, weight_decay=weight_decay)
        new_points1 = tf.squeeze(new_points1, [2])  # [B x ndataset1 x mlp[-1]]
        return new_points1
