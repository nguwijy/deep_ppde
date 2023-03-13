# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import os
import time
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.python.ops import control_flow_ops
import shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)))

tf.disable_v2_behavior()

def neural_net(y, neurons, name, is_training,
               reuse=None, decay=0.9, dtype=tf.float32):

    def batch_normalization(x):
        beta = tf.get_variable(
            'beta', [x.get_shape()[-1]], dtype,
            tf.zeros_initializer())
        gamma = tf.get_variable(
            'gamma', [x.get_shape()[-1]], dtype,
            tf.ones_initializer())
        mv_mean = tf.get_variable(
            'mv_mean', [x.get_shape()[-1]], dtype=dtype,
            initializer=tf.zeros_initializer(), trainable=False)
        mv_var = tf.get_variable(
            'mv_var', [x.get_shape()[-1]], dtype=dtype,
            initializer=tf.ones_initializer(), trainable=False)
        mean, variance = tf.nn.moments(x, [0], name='moments')
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            assign_moving_average(mv_mean, mean, decay,
                                  zero_debias=True))
        tf.add_to_collection(
            tf.GraphKeys.UPDATE_OPS,
            assign_moving_average(mv_var, variance, decay,
                                  zero_debias=False))
        mean, variance = tf.cond(is_training,
                                  lambda: (mean, variance),
                                  lambda: (mv_mean, mv_var))
        # mean, variance = utils.smart_cond(is_training,
        #                                   lambda: (mean, variance),
        #                                   lambda: (mv_mean, mv_var))
        return tf.nn.batch_normalization(x, mean, variance,
                                         beta, gamma, 1e-6)

    def layer(x, out_size, activation):
        w = tf.get_variable(
            'weights', [x.get_shape().as_list()[-1], out_size],
            dtype, tf.initializers.glorot_uniform())
            # dtype, initializers.xavier_initializer())
        return activation(batch_normalization(tf.matmul(x, w)))

    with tf.variable_scope(name, reuse=reuse):
        y = batch_normalization(y)
        for i in range(len(neurons) - 1):
            with tf.variable_scope('layer_%i_' % (i + 1)):
                y = layer(y, neurons[i], tf.nn.relu)
        with tf.variable_scope('layer_%i_' % len(neurons)):
            return layer(y, neurons[-1], tf.identity)


def splitting_model(y, t, n, phi, f, net,
                    neurons, batch_size, dtype=tf.float32):

    v_n = None

    _y = y[:, :, 1]
    if net == 0:
        v_i = phi(_y)
    else:
        v_i = neural_net(_y, neurons, 'v_%i_' % net,
                        tf.constant(False, dtype=tf.bool), dtype=dtype)
    grad_v = tf.gradients(v_i, _y)

    if net == n - 1:
        v_n = tf.get_variable('v_%i_' % (net + 1), [], dtype,
                              tf.random_uniform_initializer())
        v_j = tf.ones([batch_size, 1], dtype) * v_n
    else:
        v_j = neural_net(y[:, :, 0], neurons, 'v_%i_' % (net + 1),
                         tf.constant(True, dtype=tf.bool), dtype=dtype)

    loss = (v_j - tf.stop_gradient(v_i
            + t / n * f(_y, v_i, grad_v[0]))) ** 2

    return tf.reduce_mean(loss), v_n


def simulate(t, n, d, sde, phi, f, neurons, train_steps, batch_size,
             lr_boundaries, lr_values, path, epsilon=1e-8):

    for i in range(n):

        tf.reset_default_graph()

        y = sde(d, n - i - 1)
        loss, v_n = splitting_model(y, t, n, phi, f, i,
                                    neurons, batch_size)

        global_step = tf.get_variable(
            'global_step_%i_' % (i + 1), [], tf.int32,
            tf.zeros_initializer(), trainable=False)
        learning_rate = tf.train.piecewise_constant(
            global_step, lr_boundaries, lr_values)
        update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS, 'v_%i_' % (i + 1))
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(
                learning_rate, epsilon=epsilon).minimize(
                loss, global_step=global_step)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            var_list_n = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, 'v_%i_' % (i + 1))
            saver_n = tf.train.Saver(var_list=var_list_n)

            if i > 0:
                saver_p = tf.train.Saver(
                    var_list=tf.get_collection(
                        tf.GraphKeys.GLOBAL_VARIABLES, 'v_%i_' % i))
                saver_p.restore(
                    sess, os.path.join(path, 'model_%i_' % i))

            for _ in range(train_steps):
                sess.run(train_op)

            saver_n.save(
                sess, os.path.join(path, 'model_%i_' % (i + 1)))

            if i == n - 1:
                return sess.run(v_n)

def phi(y):
    # return tf.reduce_sum(y ** 2, axis=1, keepdims=True) ** 0.25
    temp = tf.reduce_sum(y, 1, keep_dims=True)
    return tf.where(temp>0, temp, tf.zeros_like(temp))


def f(x, y, z):
    # return -tf.reduce_sum(z ** 2, axis=1, keepdims=True)
    return 0


def sde(_d, n):
    y = [tf.ones((batch_size, _d)) * (-0.7/1)]
    for _n in range(n + 1):
        y.append(y[-1] * (1. - 0.01 * T / N
          + 0.1 * tf.random_normal((batch_size, _d),
                                         stddev=np.sqrt(T / N)))
          + 1/N)
    return tf.stack(y[n:n + 2], axis=2)

batch_size = 256
# batch_size = 64
train_steps = 600
lr_boundaries = [400, 500]
lr_values = [0.1, 0.01, 0.001]
# train_steps = 1000
# lr_boundaries = [250, 500, 750]
# lr_values = [0.1, 0.01, 0.001, 0.0001]

T = 0.1
_file = open(f'logs/ariel_asian_T_{T}.csv', 'w')
_file.write('d,T,N,run,y0,runtime\n')

# exact_sol = 0.3001998
for d in [1]:

    neurons = [d + 10, d + 10, 1]
    N = int(T / .01)

    for run in range(10):

        # path = '/tmp/hjb'
        path = os.path.join(os.getcwd(),'tmp')
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        t_0 = time.time()
        v_n = simulate(T, N, d, sde, phi, f, neurons,
                       train_steps, batch_size,
                       lr_boundaries, lr_values, path)
        t_1 = time.time()

        _file.write('%i, %f, %i, %i, %f, %f\n'
                    % (d, T, N, run, v_n, t_1 - t_0))
        _file.flush()
        print(d, T, N, run, v_n, t_1 - t_0)

_file.close()

# these temp files are annoying when I upload files to google drive
if os.path.exists(path):
    shutil.rmtree(path)
