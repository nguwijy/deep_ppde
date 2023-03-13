import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow.compat.v1 as tf
import time as ttt
import os
import pprint
import keras.backend as K
from tensorflow.compat.v1.keras.backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM
from tensorflow.keras.optimizers import Adam
from keras.regularizers import L1L2
os.chdir(os.path.dirname(os.path.abspath(__file__)))

tf.disable_v2_behavior()

def loss_function(time,path, M):

    a = tf.Variable(tf.zeros([M, n_a]), trainable=False)
    c = tf.Variable(tf.zeros([M, n_a]), trainable=False)

    ##. Approximate f(X_{t_0})
    # input_x = tf.slice(path, [0,0,0], [M,1,d])
    # input_x = tf.squeeze(input_x, axis=[1])
    input_x = path[:, 0, :]
    input_t = tf.slice(time, [0,0], [M,1])
    inputt_f = K.concatenate([input_x, input_t, a])
    f = NN(inputt_f)

    partial_x_f_list = []
    partial_xx_f_list = []

    ##. calculate space derivative
    for ii in range(d):
        delta = 0.01*inputt_f[:, ii]
        # this is to prevent zero bump_i
        bump_i = tf.where( delta>0, tf.maximum(delta, 1e-5), tf.minimum(delta, -1e-5))
        inputt_up = K.concatenate([
                        inputt_f[:,:ii],
                        1.01*tf.expand_dims(inputt_f[:,ii], -1),
                        inputt_f[:,(ii+1):] ])
        f_up = NN(inputt_up)
        partial_x_f_list.append( tf.squeeze(f_up-f, axis=[1])/bump_i )

        partial_xx_f_temp = []
        for jj in range(d):
            delta = 0.01*inputt_f[:, jj]
            # this is to prevent zero bump_j
            bump_j = tf.where( delta>0, tf.maximum(delta, 1e-5), tf.minimum(delta, -1e-5))
            inputt_down = K.concatenate([
                            inputt_f[:,:jj],
                            0.99*tf.expand_dims(inputt_f[:,jj], -1),
                            inputt_f[:,(jj+1):] ])
            f_down = NN(inputt_down)
            partial_xx_f_temp.append( tf.squeeze(f_up-2*f+f_down, axis=[1])/bump_i/bump_j )
        partial_xx_f_list.append( tf.stack(partial_xx_f_temp, axis=1) )

    partial_x_f = tf.stack(partial_x_f_list, axis=1)
    partial_xx_f = tf.stack(partial_xx_f_list, axis=2)

    ##. Next LSTM cell
    input_x_reshape = tf.reshape(input_x, (M,1,d))
    input_t_reshape = tf.reshape(input_t, (M,1,1))
    inputt = K.concatenate([input_x_reshape, input_t_reshape])
    a, _, c = LSTM_cell(inputt, initial_state = [a, c])

    ##.   calculate time derivative
    input_t_time = tf.slice(time, [0, 1], [M, 1])
    # input_t_time = K.slice(time, [0, 1], [M, 1])
    inputt_time = K.concatenate([input_x, input_t_time, a])
    f_flat = NN(inputt_time)
    partial_t_f = (f_flat - f)/dt

    xpath = tf.slice(path, [0,0,0], [M,1,d])
    Loss = tf.reduce_sum(tf.square(
            ppde(f, partial_t_f, partial_x_f, partial_xx_f, xpath) ))

    #############################################################################
    for i in range(1, steps+1): # Iterate through every timestep

        ##. Approximate f(X_{t_i})
        # input_x = tf.slice(path, [0,i,0], [M,1,d])
        # input_x = tf.squeeze(input_x, axis=[1])
        input_x = path[:, i, :]
        input_t = tf.slice(time, [0,i], [M,1])
        inputt_f = K.concatenate([input_x, input_t, a])
        f = NN(inputt_f)

        partial_x_f_list = []
        partial_xx_f_list = []

        ##. calculate space derivative
        for ii in range(d):
            delta = 0.01*inputt_f[:, ii]
            # this is to prevent zero bump_j
            bump_i = tf.where( delta>0, tf.maximum(delta, 1e-5), tf.minimum(delta, -1e-5))
            inputt_up = K.concatenate([
                            inputt_f[:,:ii],
                            1.01*tf.expand_dims(inputt_f[:,ii], -1),
                            inputt_f[:,(ii+1):] ])
            f_up = NN(inputt_up)
            partial_x_f_list.append( tf.squeeze(f_up-f, axis=[1])/bump_i )

            partial_xx_f_temp = []
            for jj in range(d):
                delta = 0.01*inputt_f[:, jj]
                # this is to prevent zero bump_j
                bump_j = tf.where( delta>0, tf.maximum(delta, 1e-5), tf.minimum(delta, -1e-5))
                inputt_down = K.concatenate([
                                inputt_f[:,:jj],
                                0.99*tf.expand_dims(inputt_f[:,jj], -1),
                                inputt_f[:,(jj+1):] ])
                f_down = NN(inputt_down)
                partial_xx_f_temp.append( tf.squeeze(f_up-2*f+f_down, axis=[1])/bump_i/bump_j )
            partial_xx_f_list.append( tf.stack(partial_xx_f_temp, axis=1) )

        partial_x_f = tf.stack(partial_x_f_list, axis=1)
        partial_xx_f = tf.stack(partial_xx_f_list, axis=2)

        ##. Next LSTM cell
        input_x_reshape = tf.reshape(input_x, (M,1,d))
        input_t_reshape = tf.reshape(input_t, (M,1,1))
        inputt = K.concatenate([input_x_reshape, input_t_reshape])
        a, _, c = LSTM_cell(inputt, initial_state = [a, c])

        ##. calculate time derivative
        input_t_time = tf.slice(time, [0, i+1], [M, 1])
        # input_t_time = K.slice(time, [0, i+1], [M, 1])
        inputt_time = K.concatenate([input_x, input_t_time, a])
        f_flat = NN(inputt_time)
        partial_t_f = (f_flat - f)/dt

        xpath = tf.slice(path, [0,0,0], [M,i+1,d])
        Loss += tf.reduce_sum(tf.square(
                ppde(f, partial_t_f, partial_x_f, partial_xx_f, xpath) ))

    #############################################################################
    ##. Terminal cost
    Loss += tf.reduce_sum( tf.square( f-phi(path) ) )*steps

    solution = f
    # return Loss/M/steps, solution, time_derivative, space_derivative, space_2nd_derivative
    return Loss/M/steps, solution


def phi(apath):
    '''
    Givin a path, it outputs the terminal condition
    '''
    if which_type == 'galerkin_control':
        reduced_x = tf.reduce_mean(apath, 2)
        # note that we use linear interpolation
        # that means the integration is in the sense of trapezoidal rule
        reduced_int = dt * ( tf.reduce_sum(reduced_x, 1) + tf.reduce_sum(reduced_x[:,1:-1], 1) ) / 2
        return tf.cos(reduced_x[:, -1] + reduced_int)
    elif which_type == 'galerkin_barrier':
        K, B = 0.7, 1.2
        # first average along the dimension axis
        reduced_x = tf.reduce_mean(apath, 2)
        reduced_term = reduced_x[:, -1] - K
        up_flag = B - tf.reduce_max(reduced_x, 1)
        return tf.where(tf.math.logical_and(up_flag > 0, reduced_term > 0), reduced_term, tf.zeros_like(reduced_term))
    elif which_type == 'galerkin_asian':
        K = 0.7
        # first average along the dimension axis
        reduced_x = tf.reduce_mean(apath, 2)
        # note that we use linear interpolation in our paper
        # that means the integration is in the sense of trapezoidal rule
        reduced_mean = dt * ( tf.reduce_sum(reduced_x, 1) + tf.reduce_sum(reduced_x[:,1:-1], 1) ) / (2*T)
        reduced_mean -= K
        return tf.where(reduced_mean>0, reduced_mean, tf.zeros_like(reduced_mean))


def ppde(u, u_t, u_x, u_xx, apath):
    '''
    Given u and all derivatives, calculate the PPDE terms
    '''
    if which_type == 'galerkin_control':
        mu_low, mu_high, sig_low, sig_high = -0.2, 0.2, 0.2, 0.3
        a_low, a_high = sig_low**2, sig_high**2

        # first sum along the dimension axis
        reduced_x = tf.reduce_mean(apath, 2)
        # note that we use linear interpolation
        # that means the integration is in the sense of trapezoidal rule
        reduced_int = dt * ( tf.reduce_sum(reduced_x, 1) + tf.reduce_sum(reduced_x[:,1:-1], 1) ) / 2
        reduced_sin = tf.sin(reduced_x[:,-1] + reduced_int)
        reduced_cos = tf.cos(reduced_x[:,-1] + reduced_int)
        reduced_ux = tf.reduce_sum(u_x,1)
        reduced_uxx = tf.linalg.trace(u_xx)
        cancel = -a_low * reduced_uxx / 2
        mu = tf.where(reduced_ux>0, mu_low*reduced_ux, mu_high*reduced_ux)
        a = tf.where(reduced_uxx>0, a_high*reduced_uxx, a_low*reduced_uxx) / 2
        small_f = tf.where(reduced_sin>0, reduced_x[:,-1] + mu_high, reduced_x[:,-1] + mu_low) \
                    * reduced_sin + 1/d * tf.where(reduced_cos>0, a_low*reduced_cos/2 , a_high*reduced_cos/2)
        return tf.squeeze(u_t) + mu + a + small_f
    elif which_type == 'galerkin_barrier' or which_type == 'galerkin_asian':
        r = 0.01
        sig = 0.1
        x = apath[:,-1,:]
        return tf.squeeze(u_t) + tf.squeeze(tf.matmul(tf.expand_dims(r*x, -1), \
                        tf.expand_dims(u_x, -1), transpose_a=True)) \
                    + 0.5 * sig**2 * tf.linalg.trace(tf.matmul(tf.linalg.diag(x**2), u_xx)) \
                    - tf.squeeze(r*u)


def generate_t(T, steps, M ):
    '''
    time discretization (M * (steps+1 +1))
    for computing the time derivative, we need an extra time step.
    '''
    t_temp = np.linspace(1e-6, T- 1e-6, steps +1, dtype = np.float32)
    return np.tile(np.concatenate((t_temp, [T + dt])), (M,1)) # extra after terminal


def init_x():
    '''
    the initial points
    '''
    if which_type == 'galerkin_control':
        return 0.0
    elif which_type == 'galerkin_barrier' or which_type == 'galerkin_asian' \
            or which_type == 'barrier_nonlinear' or which_type == 'asian_nonlinear':
        return 1.0


def b(x):
    '''
    the drift of the SDE
    '''
    if which_type == 'galerkin_control':
        return np.zeros_like(x)
    elif which_type == 'galerkin_barrier' or which_type == 'galerkin_asian':
        r = 0.01
        return r*x


def sigma(x):
    '''
    the volatility of the SDE
    '''
    if which_type == 'galerkin_control':
        sig_low = 0.2
        # return sig_low * tf.eye( _d, batch_shape = [batch_size] )
        return sig_low
    elif which_type == 'galerkin_barrier' or which_type == 'galerkin_asian':
        sig = 0.1
        return sig*x


def Create_paths(i, M):
    '''
    GBM paths (M * (steps+1)) with seed i
    number of steps could be 100, 200, 500, 1000.
    '''
    x0 = init_x()

    # generate time steps for each path
    np.random.seed(i)
    x = np.tile(x0, (M, 1, d))
    path = np.tile(x0, (M, 1, d))

    dW = np.sqrt(dt)*np.random.normal(size=(M, steps, d))
    for k in range(steps):
        x += b(x) * dt + sigma(x)* dW[:,k:k+1,:]
        path = np.concatenate((path, x), axis=1)

    return np.array(path, dtype=np.float32)

M = 256 # number of samples in a batch
T = 0.1 # terminal time
dt = 0.01 # detla_t = 0.01, 0.005, 0.002, 0.001
steps = int(T/dt) # number of time steps

# which_type = 'galerkin_asian'
# which_type = 'galerkin_barrier'
# which_type = 'galerkin_control'

# n_a = 1*128 # number of hidden neurons in the LSTM network

Epoch = 1000

for which_type in [ 'galerkin_asian', 'galerkin_barrier', 'galerkin_control' ]:

    print(which_type)
    _file = open(f'logs/{which_type}_T_{T}.csv', 'w')
    _file.write('d,T,N,run,y0,runtime\n')

    for d in [1, 10]:

        # input time and path as placeholders
        path = tf.placeholder(dtype=tf.float32, shape=[M,steps+1,d])
        time = tf.placeholder(dtype = tf.float32, shape = [M, steps +1 + 1]) # extra after T

        for run in range(10):
            with tf.Session() as sess:

                n_a = d+10 # number of hidden neurons in the LSTM network
                LSTM_cell = LSTM(n_a, return_state = True) # This is used to capture the long term dependency

                # This feedforward neural network is used to compute the derivatives.
                # input dimension is (1+1+n_a) = (space, time, path (which is characterized by n_a hidden neurons))
                NN = Sequential([
                    Dense(d+10, input_shape=(1+d+n_a,)),
                    Activation('tanh'),
                    Dense(d+10),
                    Activation('tanh'),
                    Dense(1)
                ])

                loss, solution = loss_function(time, path, M)

                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = 0.01
                # exponential decay learning rate
                learning_rate = tf.maximum(tf.train.exponential_decay(starter_learning_rate, global_step, \
                    50, 0.98, staircase=True), tf.constant(0.00001))
                # adam optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
                gvs = optimizer.compute_gradients(loss)
                capped_gvs = [(tf.clip_by_norm(grad, 5.) if grad is not None else None, var) for grad, var in gvs]
                train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

                init = tf.global_variables_initializer()
                sess.run(init)

                np.random.seed(8)

                time_feed = generate_t(T, steps, M)
                start_time = ttt.time()

                for it in range(Epoch):
                    seed = ( it % 100 ) + run
                    path_feed =  Create_paths(seed, M)
                    feed_dict = {path: path_feed, time: time_feed}
                    sess.run(train_op, feed_dict)

                elapsed = ttt.time() - start_time
                loss_value = sess.run(loss, feed_dict)
                lr = sess.run(learning_rate)
                solution_pred= sess.run([solution], feed_dict)
                print("session {}, dimension is {}, predicted solution is {}, loss is {}, and learning rate is {}, \
                        elapsed is {}.\n".format(run, d, solution_pred[0][0][0], loss_value, lr, elapsed))

                _file.write('%i, %f, %i, %i, %f, %f\n'
                            % (d, T, steps, run, solution_pred[0][0][0], elapsed))
                _file.flush()
