import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))


def discriminative_loss_single(prediction, correct_label, feature_dim, delta_v, delta_d, param_var, param_dist):
    ''' Discriminative loss for a single prediction/label pair.
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param label_shape: shape of label
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    '''

    ### Reshape so pixels are aligned along a vector
    # correct_label = tf.reshape(correct_label, [label_shape[1] * label_shape[0]])
    reshaped_pred = tf.reshape(prediction, [-1, feature_dim])

    ### Count instances
    unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)

    counts = tf.cast(counts, tf.float32)
    num_instances = tf.size(unique_labels)

    segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)

    mu = tf.div(segmented_sum, tf.reshape(counts, (-1, 1)))
    # partitions = tf.reduce_sum(tf.one_hot(unique_id, tf.shape(mu)[0], dtype=tf.int32), 0)
    # mu_expand = tf.dynamic_partition(mu, partitions, 2)
    mu_expand = tf.gather(mu, unique_id)

    ### Calculate l_var
    # distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    # tmp_distance = tf.subtract(reshaped_pred, mu_expand)
    tmp_distance = reshaped_pred - mu_expand
    distance = tf.norm(tmp_distance, ord=1, axis=1)

    distance = tf.subtract(distance, delta_v)
    distance = tf.clip_by_value(distance, 0., distance)
    distance = tf.square(distance)

    l_var = tf.unsorted_segment_sum(distance, unique_id, num_instances)
    l_var = tf.div(l_var, counts)
    l_var = tf.reduce_sum(l_var)
    l_var = tf.divide(l_var, tf.cast(num_instances, tf.float32))

    ### Calculate l_dist

    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3

    mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
    mu_band_rep = tf.tile(mu, [1, num_instances])
    mu_band_rep = tf.reshape(mu_band_rep, (num_instances * num_instances, feature_dim))

    mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

    # Filter out zeros from same cluster subtraction
    eye = tf.eye(num_instances)
    zero = tf.zeros(1, dtype=tf.float32)
    diff_cluster_mask = tf.equal(eye, zero)
    diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])
    mu_diff_bool = tf.boolean_mask(mu_diff, diff_cluster_mask)

    mu_norm = tf.norm(mu_diff_bool, ord=1, axis=1)
    mu_norm = tf.subtract(2. * delta_d, mu_norm)
    mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
    mu_norm = tf.square(mu_norm)

    l_dist = tf.reduce_mean(mu_norm)

    def rt_0(): return 0.

    def rt_l_dist(): return l_dist

    l_dist = tf.cond(tf.equal(1, num_instances), rt_0, rt_l_dist)

    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist

    loss = param_scale * (l_var + l_dist)

    return loss, l_var, l_dist


def discriminative_loss(prediction, correct_label, feature_dim, delta_v, delta_d, param_var, param_dist):
    ''' Iterate over a batch of prediction/label and cumulate loss
    :return: discriminative loss and its two components
    '''

    def cond(label, batch, out_loss, out_var, out_dist, i):
        return tf.less(i, tf.shape(batch)[0])

    def body(label, batch, out_loss, out_var, out_dist, i):
        disc_loss, l_var, l_dist = discriminative_loss_single(prediction[i], correct_label[i], feature_dim, delta_v, delta_d, param_var, param_dist)

        out_loss = out_loss.write(i, disc_loss)
        out_var = out_var.write(i, l_var)
        out_dist = out_dist.write(i, l_dist)

        return label, batch, out_loss, out_var, out_dist, i + 1

    # TensorArray is a data structure that support dynamic writing
    output_ta_loss = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_var = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    output_ta_dist = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    _, _, out_loss_op, out_var_op, out_dist_op, _ = tf.while_loop(cond, body, [correct_label, prediction, output_ta_loss, output_ta_var, output_ta_dist, 0])
    out_loss_op = out_loss_op.stack()
    out_var_op = out_var_op.stack()
    out_dist_op = out_dist_op.stack()

    disc_loss = tf.reduce_mean(out_loss_op)
    l_var = tf.reduce_mean(out_var_op)
    l_dist = tf.reduce_mean(out_dist_op)

    return disc_loss, l_var, l_dist
