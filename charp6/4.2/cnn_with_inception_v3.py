# -*- coding: utf-8 -*-
"""
使用TenforFlow-Slim工具来实现一个卷积层
"""
import tensorflow as tf
import tensorflow.contrib.slim

# 直接使用TensorFlow原始API实现卷积层。
with tf.variable_scope("scope_name"):
    weights = tf.get_variable("weights", ...)
    biases = tf.get_variable("bias", ...)
    conv = tf.nn.conv2d(...);
relu = tf.nn.relu(tf.nn.bias_add(conv, biases))

# 使用TensorFlow-Slim实现卷积层。通过TenforFlow-Slim可以在一行中实现一个卷积层
# 的前向传播算法。slim.conv2d函数有三个参数是必填的。第一个参数为输入节点矩阵，第
# 二参数是当前卷积层过滤器的深度，第三个参数是过滤器的尺寸。可选的参数有过滤器移动的步
# 长、是否使用全0填充、激活函数的选择以及变量的命名空间等。
net = slim.conv2d(input, 32, [3, 3])

# slim.arg_scope函数可以用于设置默认的参数取值。slim.arg_scope函数的第一个参数是
# 一个函数列表，在这个列表中的函数将使用默认的参数取值。比如通过下面的定义，调用
# slim.conv2d(net, 320, [1, 1])函数时会自动加上stride=1和padding='SAMEde1canshu1
# 。如果在函数调用时指定了stride，那么这里设置的默认值就不会再使用。通过这种方式
# 可以进一步减少冗余的代码。
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                    stride=1, padding='SAME'):
    pass
    # 此处省略了Inception-v3模型中其他的网络结构而直接实现后面红色方框中的
    # Inception结构。假设输入图片经过之前的神经网络前向传播的结果保存在变量net
    # 中
    net = 上一层的输出节点矩阵