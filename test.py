# -*- coding: utf-8 -*-
"""
本模块主要做一些函数功能的测试工作
"""
import tensorflow as tf

a = tf.constant([[[[1, -1, 0], [-1, 2, 1], [0, 2, -2]]]], dtype="float32")
w = tf.constant([[1, -1], [0, 2]], dtype="float32")
conv = tf.nn.conv2d(a, w, strides=[1, 1, 1, 1], padding='SAME')

with tf.session() as sess:
    print (sess.run(conv))