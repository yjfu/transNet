import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import ConfigProto
from datetime import datetime
import os

def kernel_with_weight_decay(shape, name, weight_decay=0.0002):
    with tf.device('/cpu:0'):
        weight = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.001))
    decay = tf.multiply(tf.nn.l2_loss(weight), weight_decay, name="weight_decay")
    tf.add_to_collection("losses", decay)
    return weight

def bias(shape, name):
    with tf.device('/cpu:0'):
        _bias = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
    return _bias

def transNet(input):
    """
    Define network structure
    :param input: data with 6 channel, 3 for rgb, 1 for predict result from last frame,
                and 2 for optical flow result
    :return: predict result
    """
    with tf.variable_scope("TransNet", values=[input]):
        with tf.variable_scope("conv1_1"):
            _kernel = kernel_with_weight_decay(shape=[3, 3, 6, 32],
                                              name='kernel')
            _bias = bias(shape=[32], name='bias')
            _conv = tf.nn.atrous_conv2d(value=input, filters=_kernel, rate=1,
                                        padding='SAME')
            _pre_activation = tf.nn.bias_add(_conv, _bias)
            conv1_1 = tf.nn.relu(_pre_activation)
        with tf.variable_scope("conv1_2"):
            _kernel = kernel_with_weight_decay(shape=[3, 3, 32, 32],
                                               name='kernel')
            _bias = bias(shape=[32], name='bias')
            _conv = tf.nn.atrous_conv2d(value=conv1_1, filters=_kernel, rate=1,
                                        padding='SAME')
            _pre_activation = tf.nn.bias_add(_conv, _bias)
            conv1_2 = tf.nn.relu(_pre_activation)
        with tf.variable_scope("conv2_1"):
            _kernel = kernel_with_weight_decay(shape=[3, 3, 32, 64],
                                               name='kernel')
            _bias = bias(shape=[64], name='bias')
            _conv = tf.nn.atrous_conv2d(value=conv1_2, filters=_kernel, rate=2,
                                        padding='SAME')
            _pre_activation = tf.nn.bias_add(_conv, _bias)
            conv2_1 = tf.nn.relu(_pre_activation)
        with tf.variable_scope("conv2_2"):
            _kernel = kernel_with_weight_decay(shape=[3, 3, 64, 64],
                                               name='kernel')
            _bias = bias(shape=[64], name='bias')
            _conv = tf.nn.atrous_conv2d(value=conv2_1, filters=_kernel, rate=2,
                                        padding='SAME')
            _pre_activation = tf.nn.bias_add(_conv, _bias)
            conv2_2 = tf.nn.relu(_pre_activation)
        with tf.variable_scope("conv3_1"):
            _kernel = kernel_with_weight_decay(shape=[3, 3, 64, 128],
                                               name='kernel')
            _bias = bias(shape=[128], name='bias')
            _conv = tf.nn.atrous_conv2d(value=conv2_2, filters=_kernel, rate=3,
                                        padding='SAME')
            _pre_activation = tf.nn.bias_add(_conv, _bias)
            conv3_1 = tf.nn.relu(_pre_activation)
        with tf.variable_scope("conv3_2"):
            _kernel = kernel_with_weight_decay(shape=[3, 3, 128, 128],
                                               name='kernel')
            _bias = bias(shape=[128], name='bias')
            _conv = tf.nn.atrous_conv2d(value=conv3_1, filters=_kernel, rate=3,
                                        padding='SAME')
            _pre_activation = tf.nn.bias_add(_conv, _bias)
            conv3_2 = tf.nn.relu(_pre_activation)
        with tf.variable_scope("conv4_1"):
            _kernel = kernel_with_weight_decay(shape=[3, 3, 128, 32],
                                               name='kernel')
            _bias = bias(shape=[32], name='bias')
            _conv = tf.nn.atrous_conv2d(value=conv3_2, filters=_kernel, rate=3,
                                        padding='SAME')
            _pre_activation = tf.nn.bias_add(_conv, _bias)
            conv4_1 = _pre_activation
        with tf.variable_scope("conv4_2"):
            _kernel = kernel_with_weight_decay(shape=[1, 1, 32, 1],
                                               name='kernel')
            _bias = bias(shape=[1], name='bias')
            _conv = tf.nn.conv2d(input=conv4_1, filter=_kernel, strides=[1, 1, 1, 1],
                                 padding='SAME')
            _pre_activation = tf.nn.bias_add(_conv, _bias)
            predict_result = _pre_activation

    return predict_result

def calculate_balance_loss(predict, label):
    """
    Calculate the loss according not only accuracy but also the range of likelihood the
    network predicted, i.e. if the predicted result is too bigger(or, too small a negative),
    the loss will be greater.
    And note that this loss function will keep a balance of the proportion of foreground
    and background.
    :param predict: the predicted likelihood produced by network
    :param label: the ground truth
    :return: a scalar loss
    """

    # predict is the probability of pixels, which should be cast to binary result(0 or 1)
    binary_predict = tf.cast(tf.greater_equal(predict, 0), tf.float32)
    # make pixel-wise loss map, which will be calculated not only according to accuracy,
    # (which is calculated by the former), but also the range, i.e will be more negative if
    # |predict| is bigger(with following op, a smaller negative will become a bigger positive)
    loss_val = tf.multiply(predict, (label - binary_predict)) - tf.log(
        1 + tf.exp(predict - 2 * tf.multiply(predict, binary_predict)))

    # if the predict of one pixel is wrong, then loss will be -loss_val(a positive number)
    loss_pos = tf.reduce_sum(-tf.multiply(label, loss_val))
    loss_neg = tf.reduce_sum(-tf.multiply(1.0 - label, loss_val))
    # calculate the proportion of object (which is positive sample here) and background
    # (which is negative sample) so that we can take a balance of the effect of two
    # part of data
    pos_pix_num = tf.reduce_sum(label)
    neg_pix_num = tf.reduce_sum(1-label)
    pos_proportion = pos_pix_num/(neg_pix_num+pos_pix_num)
    neg_proportion = 1-pos_proportion

    loss = pos_proportion*loss_neg+neg_proportion*loss_pos
    return loss

def calculate_simple_loss(predict, label):
    """
    Calculate the simple loss which only according to the pixel-wise accuracy
    :param predict: the predicted likelihood produced by network
    :param label: the ground truth
    :return: a scalar loss
    """

    # predict is the probability of pixels, which should be cast to binary result(0 or 1)
    predict_bool = tf.greater_equal(predict, 0)
    label_bool = tf.greater_equal(label, 0.5)
    error_map = tf.logical_xor(label_bool, predict_bool)
    pixel_num = tf.reduce_sum(label+1)-tf.reduce_sum(label)
    loss = tf.reduce_sum(tf.cast(error_map, tf.float32))/pixel_num

    return loss
def calculate_IOU_score(predict, label):
    """
    Calculate the IOU score
    :param predict: the predicted likelihood produced by network
    :param label: the ground truth
    :return: IOU score
    """

    # predict is the probability of pixels, which should be cast to binary result(0 or 1)
    predict_bool = tf.greater_equal(predict, 0)
    label_bool = tf.greater_equal(label, 0.5)
    union = tf.logical_or(predict_bool, label_bool)
    intersection = tf.logical_and(predict_bool, label_bool)
    u = tf.reduce_sum(tf.cast(union, tf.float32))
    i = tf.reduce_sum(tf.cast(intersection, tf.float32))
    return i/u

def train(dataset, learning_rate, momentum, global_step, logs_path,
          save_step, max_train_iters, display_step, ckpt_name):
    """
    Train the net from nearest checkpoint(if have)
    :param dataset:
    :param learning_rate:
    :param momentum:
    :param global_step:
    :param logs_path: dir where the log will be wrote in
    :param save_step: model will be saved every save_step step
    :param max_train_iters: training will stop when global_step is equal to this param
    :param display_step: print some info e.g.IOU,losses... every display_step step
    :param ckpt_name: model will saved in logs_path/ckpt_name.ckpt-global_step
    :return:
    """

    # set config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    input_image = tf.placeholder(tf.float32, [dataset.batch_size, None, None, 6])
    input_label = tf.placeholder(tf.float32, [dataset.batch_size, None, None, 1])

    predict_result = transNet(input=input_image)

    with tf.name_scope("loss"):
        loss = calculate_balance_loss(predict=predict_result, label=input_label)
        tf.add_to_collection("losses", loss)
        total_loss = tf.add_n(tf.get_collection("losses"))
        IOU = calculate_IOU_score(predict=predict_result, label=input_label)
        simple_loss = calculate_simple_loss(predict=predict_result, label=input_label)
        tf.summary.scalar("total_loss", total_loss)
        tf.summary.scalar("IOU", IOU)
        tf.summary.scalar("simple_loss", simple_loss)
    with tf.name_scope("optimization"):
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=momentum)
        gradients = optimizer.compute_gradients(total_loss)
        apply_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                             global_step=global_step)

    merge_summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    # start to train
    with tf.Session(config=config) as sess:
        sess.run(init)

        summary_writer = tf.summary.FileWriter(logdir=logs_path,
                                               graph=tf.get_default_graph())
        saver = tf.train.Saver(max_to_keep=None)

        last_ckpt_path = tf.train.latest_checkpoint(checkpoint_dir=logs_path)
        if last_ckpt_path is not None:
            print "resume from checkpoint ", last_ckpt_path
            saver.restore(sess=sess, save_path=last_ckpt_path)
            step = global_step.eval()+1
        else:
            step = 1

        print "start to train"
        while step < max_train_iters+1:
            print "step %d" % step
            data, label = dataset.next_batch()
            result = sess.run([total_loss, IOU, simple_loss, merge_summary, apply_op],
                              feed_dict={input_image: data, input_label: label})
            # sess.run(apply_op)
            summary_writer.add_summary(result[3], step)

            # print result after some step
            if step % display_step == 0:
                print "{} step {}:\nTotal Losses: {:.4f}, IOU: {:.4f}," \
                      " Naive Accuracy: {:.4f}\n".format(datetime.now(), step, *result[0:3])

            # save model into ckpt
            if step % save_step == 0:
                print "save model..."
                save_path = saver.save(sess=sess,
                                       save_path=os.path.join(logs_path, ckpt_name+".ckpt"),
                                       global_step=global_step)
                print "model saved in %s" % save_path

            step += 1
        # final save
        if (step - 1) % save_step != 0:
            save_path = saver.save(sess=sess,
                                   save_path=os.path.join(logs_path, ckpt_name + ".ckpt"),
                                   global_step=global_step)
            print "model saved in %s" % save_path

        print "finish"


