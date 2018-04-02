import os
import tensorflow as tf
from data_set import DataSet
import transNet

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
# create dataset
dataset = DataSet("data_path.txt", batch_size=16)
# set a piece-wise learning rate
ini_learning_rate = 1e-7
boundaries = [10000, 15000, 25000, 35000]
lr_values = [ini_learning_rate, ini_learning_rate*0.5, ini_learning_rate*0.1,
             ini_learning_rate*0.05, ini_learning_rate*0.01]
# set momentum
momentum = 0.9
# title is also the ckpt_name
title = "naive-transNet"
root_dir = os.path.realpath('__file__')
logs_path = os.path.join(root_dir, "models", title)
# set save_step and display_step
save_step = 5
display_step = 1
# set train iteration times
max_train_iters = 10

with tf.Graph().as_default():
    with tf.device("/cpu:0"):
        global_step = tf.Variable(initial_value=0, trainable=False, name="global_step")
        learning_rate = tf.train.piecewise_constant(global_step, boundaries=boundaries,
                                                    values=lr_values)
        transNet.train(dataset=dataset, learning_rate=learning_rate, momentum=momentum,
                       global_step=global_step, logs_path=logs_path, save_step=save_step,
                       max_train_iters=max_train_iters, display_step=display_step, ckpt_name=title)



