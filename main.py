import os

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))


########## Section neural network ####################
class FullModel(tf.keras.Model):
    """
    Full Model that contains the list of all neural networks
    and the initial y0, z0, g0
    """
    def __init__(self, config, eqn):
        super(FullModel, self).__init__()
        self.config = config
        self.eqn = eqn
        self.use_lstm = config.use_lstm
        self.lstm_net = [
            tf.keras.layers.LSTM(self.config.dim, return_sequences=True, return_state=True) for _ in range(3)
        ]
        self.subnet = [FeedForwardSubNet(config, eqn)
                      for _ in range(config.N-1)]
        y0 = tf.Variable(tf.random_uniform_initializer()(
                          shape=[1,1], dtype=config.dtype))
        if self.eqn.ppde_type == 'linear':
            self.subnet.append([y0])
        elif self.eqn.ppde_type == 'semilinear':
            z0 = tf.Variable(tf.random_uniform_initializer()(
                          shape=[1,config.dim], dtype=config.dtype))
            self.subnet.append([y0, z0])
        elif self.eqn.ppde_type == 'fully_nonlinear':
            z0 = tf.Variable(tf.random_uniform_initializer()(
                          shape=[1,config.dim], dtype=config.dtype))
            g0 = tf.Variable(tf.random_uniform_initializer()(
                          shape=[1,config.dim**2], dtype=config.dtype))
            self.subnet.append([y0, z0, g0])

    def __call__(self, x, training, nn_type, idx):
        if self.use_lstm:
            # LSTM net maps batch_size x time_steps x dim -> batch_size x time_steps x dim
            # But we are only interested in the output of last time step
            nn_type_idx = {'y': 0, 'z': 1, 'g': 2}[nn_type]
            lstm_hidden = self.lstm_net[nn_type_idx](x)[0][:, -1, :]
            x = tf.concat([x[:, -1, :], lstm_hidden], axis=1)
        if nn_type=='y':
            if idx==(self.config.N-1):
                return self.subnet[idx][0]
            else:
                return self.subnet[idx](x, training, 0)
        elif nn_type=='z':
            if idx==(self.config.N-1):
                return self.subnet[idx][1]
            else:
                return self.subnet[idx](x, training, 1)
        elif nn_type=='g':
            if idx==(self.config.N-1):
                return self.subnet[idx][2]
            else:
                return self.subnet[idx](x, training, 2)

    def build(self, idx):
        """
        Make sure the model is run at least once
        before passing to the graph generated by tf.function
        """
        # the last model is not NN, hence no initialization is needed
        if idx==(self.config.N-1): return

        if self.use_lstm:
            lstm_size = self.config.dim
            x = tf.zeros([1, 1, lstm_size])
            self.lstm_net[0](x)
            self.lstm_net[1](x)
            self.lstm_net[2](x)
            fnn_size = 2 * self.config.dim
        else:
            fnn_size = self.config.dim*(self.config.N-idx)
        x = tf.zeros([1, fnn_size])
        self.subnet[idx](x, False, 0)

        if self.eqn.ppde_type == 'semilinear' or \
              self.eqn.ppde_type == 'fully_nonlinear':
            self.subnet[idx](x, False, 1)

        if self.eqn.ppde_type == 'fully_nonlinear':
            self.subnet[idx](x, False, 2)


class FeedForwardSubNet(tf.keras.Model):
    """Implementation of individual neural networks."""
    def __init__(self, config, eqn):
        super(FeedForwardSubNet, self).__init__()
        self.config = config
        self.eqn = eqn
        self.bn_layers = [[tf.keras.layers \
                             .BatchNormalization(dtype=config.dtype)
                           for _ in range(len(config.y_neurons)+1)]]
        self.dense_layers = [[tf.keras.layers.Dense(
                                                config.y_neurons[i],
                                                dtype=config.dtype,
                                                use_bias=False,
                                                activation=None)
                              for i in range(len(config.y_neurons))]]

        # Z network for the semilinear or fully nonlinear PPDE
        if self.eqn.ppde_type == 'semilinear' \
                or self.eqn.ppde_type == 'fully_nonlinear':
            self.bn_layers.append([
                tf.keras.layers.BatchNormalization(dtype=config.dtype)
                    for _ in range(len(config.z_neurons) + 1)])
            self.dense_layers.append([
                tf.keras.layers.Dense(config.z_neurons[i],
                                      dtype=config.dtype,
                                      use_bias=False,
                                      activation=None)
                    for i in range(len(config.z_neurons))])

        # Gamma network for the fully nonlinear PPDE
        if self.eqn.ppde_type == 'fully_nonlinear':
            self.bn_layers.append([
                tf.keras.layers.BatchNormalization(dtype=config.dtype)
                    for _ in range(len(config.g_neurons) + 1)])
            self.dense_layers.append([
                tf.keras.layers.Dense(config.g_neurons[i],
                                      dtype=config.dtype,
                                      use_bias=False,
                                      activation=None)
                    for i in range(len(config.g_neurons))])

    def __call__(self, x, training, nn_type):
        """
        bn -> (dense->bn->relu)*(depth-1) -> dense -> bn
        """
        x = tf.reshape(x, [x.shape[0], -1])
        x = self.bn_layers[nn_type][0](x, training)
        depth = len(self.dense_layers[nn_type])
        for i in range(depth):
            x = self.dense_layers[nn_type][i](x)
            x = self.bn_layers[nn_type][i+1](x, training)
            if i < (depth-1):
                x = tf.nn.relu(x)
        return x


########## Section equation ####################
class Equation:
    """Base class for defining the problem."""
    def __init__(self, config):
        self.config = config
        self.x_init = None

    def b(self, x):
        """Drift of the forward SDE."""
        raise NotImplementedError

    def sigma(self, x):
        """Diffusion of the forward SDE."""
        raise NotImplementedError

    def sigma_inverse(self, x):
        """Inverse of the diffusion of the forward SDE."""
        # if this function is not implemented,
        # we use inv function in tf.linalg
        return tf.linalg.inv(self.sigma(x))

    def sde(self, n):
        """Simulate forward SDE."""
        x = [self.x_init*tf.ones((self.config.batch_size,
                self.config.dim), dtype=self.config.dtype)]
        dw = []
        for _ in range(n + 1):
            dw.append(
                tf.random.normal(
                    (self.config.batch_size, self.config.dim),
                    stddev=self.config.sqrt_delta_t,
                    dtype=self.config.dtype
                )
            )
            # squeeze used because matmul produces dim of d*1
            # while the rest are d
            x.append(
                x[-1]
                + self.b(x[-1]) * self.config.delta_t
                + tf.squeeze(tf.matmul(self.sigma(x[-1]), tf.expand_dims(dw[-1], -1)), -1)
            )
        x = tf.stack(x, axis=1)
        dw = tf.stack(dw, axis=1)
        return x, dw

    def f(self, x, y, z, gamma):
        """Generator of the PPDE."""
        raise NotImplementedError

    def phi(self, x):
        """Terminal condition of the PPDE."""
        raise NotImplementedError

    def symmetrise(self, gamma):
        # reshape to get the correct [d, d] dimension
        # then symmetrize the matrix using upper triangular part
        # because the Hessian matrix is a symmetric matrix
        gamma = tf.reshape(gamma, [-1,self.config.dim,self.config.dim])
        gamma = tf.linalg.band_part(gamma, 0, -1)
        gamma = 0.5 * (gamma + tf.transpose(gamma, perm=[0, 2, 1]))
        return gamma


class ControlProblem(Equation):
    """Example in Section 5.1"""
    def __init__(self, config):
        super(ControlProblem, self).__init__(config)
        self.ppde_type = 'fully_nonlinear'   # fully nonlinear PPDE
        self.x_init = 0
        self.mu_low = -0.2
        self.mu_high = 0.2
        self.sig_low = 0.2
        self.sig_high = 0.3
        self.a_low = self.sig_low ** 2
        self.a_high = self.sig_high ** 2

    def b(self, x):
        return tf.zeros_like(x)

    def sigma(self, x):
        return self.sig_low * tf.eye(self.config.dim,
                batch_shape=[self.config.batch_size])

    def sigma_inverse(self, x):
        return tf.eye(self.config.dim,
                  batch_shape=[self.config.batch_size]) / self.sig_low

    def f(self, x, y, z, gamma):
        # sum along the dimension axis
        reduced_x = tf.reduce_mean(x, 2)
        # trapezoidal rule
        reduced_int = self.config.delta_t * (tf.reduce_sum(reduced_x,1)
                        + tf.reduce_sum(reduced_x[:,1:-1], 1)) / 2
        reduced_sin = tf.sin(reduced_x[:,-1] + reduced_int)
        reduced_cos = tf.cos(reduced_x[:,-1] + reduced_int)
        reduced_z = tf.reduce_sum(z,1)
        reduced_gamma = tf.linalg.trace(gamma)
        cancel = -self.a_low * reduced_gamma / 2
        mu = tf.where(reduced_z>0, self.mu_low*reduced_z,
                self.mu_high*reduced_z)
        a = tf.where(reduced_gamma>0, self.a_high*reduced_gamma,
                self.a_low*reduced_gamma) / 2
        small_f = reduced_sin * tf.where(reduced_sin>0,
                                reduced_x[:,-1] + self.mu_high,
                                reduced_x[:,-1] + self.mu_low) \
                    + tf.where(reduced_cos>0,
                            self.a_low*reduced_cos/2,
                            self.a_high*reduced_cos/2) / self.config.dim
        return tf.expand_dims(cancel+mu+a+small_f, -1)

    def phi(self, x):
        # average along the dimension axis
        reduced_x = tf.reduce_mean(x, 2)
        # trapezoidal rule
        reduced_int = self.config.delta_t * (tf.reduce_sum(reduced_x,1)
                        + tf.reduce_sum(reduced_x[:,1:-1], 1)) / 2
        return tf.expand_dims(tf.cos(reduced_x[:, -1]
                + reduced_int), -1)


class AsianOption(Equation):
    """Example in Section 5.2"""
    def __init__(self, config):
        super(AsianOption, self).__init__(config)
        self.ppde_type = 'linear'   # linear PPDE
        self.x_init = 1
        self.sig = 0.1
        self.r = 0.01
        self.K = 0.7

    def b(self, x):
        return self.r*x

    def sigma(self, x):
        return self.sig * tf.linalg.diag(x)

    def sigma_inverse(self, x):
        return tf.linalg.diag(tf.reciprocal(x)) / self.sig

    def f(self, x, y, z, gamma):
        return -self.r*y

    def phi(self, x):
        # average along the dimension axis
        reduced_x = tf.reduce_mean(x, 2)
        # trapezoidal rule
        reduced_mean = self.config.delta_t*(tf.reduce_sum(reduced_x,1)\
                + tf.reduce_sum(reduced_x[:,1:-1],1))/(2*self.config.T)
        reduced_mean -= self.K
        return tf.expand_dims(tf.where(reduced_mean>0,
            reduced_mean, tf.zeros_like(reduced_mean)), -1)


class BarrierOption(Equation):
    """Example in Section 5.3"""
    def __init__(self, config):
        super(BarrierOption, self).__init__(config)
        self.ppde_type = 'linear'   # linear PPDE
        self.x_init = 1
        self.sig = 0.1
        self.r = 0.01
        self.K = 0.7
        self.B = 1.2

    def b(self, x):
        return self.r*x

    def sigma(self, x):
        return self.sig * tf.linalg.diag(x)

    def sigma_inverse(self, x):
        return tf.linalg.diag(tf.reciprocal(x)) / self.sig

    def f(self, x, y, z, gamma):
        return -self.r*y

    def phi(self, x):
        # average along the dimension axis
        reduced_x = tf.reduce_mean(x, 2)
        reduced_term = reduced_x[:, -1] - self.K
        up_flag = self.B - tf.reduce_max(reduced_x, 1)
        return tf.expand_dims(tf.where(tf.math.logical_and(up_flag > 0, reduced_term > 0), reduced_term,
            tf.zeros_like(reduced_term)), -1)


########## Section PPDE solver ####################
class PPDESolver:
    """
    Define the relationship between the variables of FullModel
    according to the type of Equation.
    """
    def __init__(self, config, eqn):
        self.config = config
        self.eqn = eqn
        self.model = FullModel(config, eqn)
        self.v0 = None
        # self.idx = None
        self.lr_schedule = tf.keras \
                             .optimizers \
                             .schedules \
                             .PiecewiseConstantDecay(
                                 config.lr_boundaries,config.lr_values)
        self.epsilon = 1e-8
        self.optimizer = None

    def plt_stats(self):
        x, _ = self.eqn.sde(self.config.N - 1)

        stats = {
            'mean': np.mean,
            'std' : np.std,
        }
        for name, fun in stats.items():
            nn       = [fun(self.model(x[:, :-(idx + 1), :], False, 'y', idx).numpy()) for idx in range(self.config.N - 1)]
            terminal = [fun(self.eqn.phi(x[:, :(self.config.N + 1 - idx), :]).numpy()) for idx in range(self.config.N - 1)]
            tt       = [self.config.T - self.config.delta_t * idx for idx in range(self.config.N - 1)]
            pd.DataFrame({'t': tt, 'nn(x[:t])': nn, 'phi(x[:t])': terminal}).plot(x='t', y=['nn(x[:t])', 'phi(x[:t])'])
            plt.title(f'{self.config.eqn_name}: {name} of output with batch size {len(x)}.')
            plt.show()

    def train(self):
        for idx in range(self.config.N):
            # generate new graph using new idx with tf.function
            # this significantly speeds up the computation
            tf_train_step = tf.function(self.train_step)
            # self.idx = idx
            self.optimizer = tf.keras \
                               .optimizers \
                               .Adam(learning_rate=self.lr_schedule,
                                      epsilon=self.epsilon)
            self.model.build(idx)
            for step in range(self.config.train_steps):
                loss, v0 = tf_train_step(idx)

            del tf_train_step   # throw away old graph
        self.v0 = tf.reduce_mean(v0).numpy()   # v0 is 1x1 tensor

    def train_step(self, idx):
        x, dw = self.eqn.sde(self.config.N-idx-1)
        if idx == 0:
            trainable_var = self.model.subnet[idx].trainable_variables + self.model.lstm_net.trainable_variables
        else:
            trainable_var = self.model.subnet[idx].trainable_variables
        with tf.GradientTape(persistent=True) as tape:
            loss, v0 = self.loss_fn(x, dw[:, -1, :], idx, training=True)
        grads = tape.gradient(loss, trainable_var)
        del tape

        self.optimizer.apply_gradients(
                  (grad,var) for (grad,var) in zip(grads,trainable_var)
                  if grad is not None)
        return loss, v0

    def loss_fn(self, x, dw, idx, training):
        v0 = None   # the solution of interest defined in advance

        ### loss regarding y network ######
        if idx == 0:   # at terminal time
            y_target = self.eqn.phi(x)
        else:
            y_temp = self.model(x, False, 'y', idx-1)
            # fully nonlinear problem requires 3 networks
            if self.eqn.ppde_type == 'fully_nonlinear':
                z_temp = self.model(x, False, 'z', idx-1)
                g_temp = self.model(x, False, 'g', idx-1)
                g_temp = self.eqn.symmetrise(g_temp)
                y_target = y_temp + self.config.delta_t \
                            * self.eqn.f(x, y_temp, z_temp, g_temp)
            # semilinear problem requires 2 networks
            elif self.eqn.ppde_type == 'semilinear':
                z_temp = self.model(x, False, 'z', idx-1)
                y_target = y_temp + self.config.delta_t \
                            * self.eqn.f(x, y_temp, z_temp, z_temp)
            # linear problem requires only 1 networks
            else:
                y_target = y_temp + self.config.delta_t \
                            * self.eqn.f(x, y_temp, y_temp, y_temp)

        # the solution v0 should be fixed throughout the batch
        if idx == self.config.N-1:
            xnow = tf.slice(x, [0,0,0], [1,1,self.config.dim])
            y0 = self.model(xnow, training, 'y', idx)
            y_now = tf.tile(y0, [self.config.batch_size, 1])
            if self.eqn.ppde_type == 'linear':
                v0 = y0 + self.config.delta_t \
                            * self.eqn.f(xnow, y0, y0, y0)
        else:
            y_now = self.model(x[:, :-1, :], training, 'y', idx)

        loss = tf.reduce_mean((y_now-tf.stop_gradient(y_target)) ** 2)
        # we are done here for linear model
        if self.eqn.ppde_type == 'linear':
            return loss, v0

        ### loss regarding z network ######
        sig_inverse = self.eqn.sigma_inverse(x[:, -2, :])
        # expand_dims is needed for tensor multiplication
        # because y_target is 1d but dw is nd
        if self.config.var_reduction:
            # minus y_now for variance reduction
            z_target = tf.expand_dims(y_target - y_now, -1) \
                        * tf.matmul(sig_inverse,
                                tf.expand_dims(dw, -1),
                                transpose_a=True) \
                        / self.config.delta_t
        else:
            z_target = tf.expand_dims(y_target, -1) \
                        * tf.matmul(sig_inverse,
                                tf.expand_dims(dw, -1),
                                transpose_a=True) \
                        / self.config.delta_t

        # the solution v0 should be fixed throughout the batch
        if idx == self.config.N-1:
            xnow = tf.slice(x, [0,0,0], [1,1,self.config.dim])
            z0 = self.model(xnow, training, 'z', idx)
            z_now = tf.tile(z0, [self.config.batch_size, 1])
            if self.eqn.ppde_type == 'semilinear':
                v0 = y0 + self.config.delta_t \
                            * self.eqn.f(xnow, y0, z0, z0)
        else:
            z_now = self.model(x[:, :-1, :], training, 'z', idx)

        loss += tf.reduce_mean((z_now
                    - tf.stop_gradient(tf.squeeze(z_target,-1))) ** 2)
        # we are done here for semilinear model
        if self.eqn.ppde_type == 'semilinear':
            return loss, v0

        ### loss regarding g network ######
        dw2 = tf.matmul(tf.expand_dims(dw, -1),
                tf.expand_dims(dw, -1), transpose_b=True)
        if self.config.var_reduction:
            g_target = (tf.expand_dims(y_target-y_now,-1) \
                        - tf.matmul(
                            tf.matmul(self.eqn.sigma(x[:, -2, :]),
                              tf.expand_dims(z_now, -1),
                              transpose_a=True ),
                            tf.expand_dims(dw, -1),
                            transpose_a=True)) \
                    * tf.matmul(
                        tf.matmul(sig_inverse,
                          (dw2 - self.config.delta_t
                           * tf.eye(self.config.dim,
                              batch_shape=[self.config.batch_size])),
                           transpose_a=True),
                      sig_inverse) / (self.config.delta_t)**2
        else:
            g_target = tf.expand_dims(y_target,-1) \
                       * tf.matmul(
                         tf.matmul(sig_inverse,
                            (dw2 - self.config.delta_t
                             * tf.eye(self.config.dim,
                                batch_shape=[self.config.batch_size])),
                             transpose_a=True),
                         sig_inverse) / (self.config.delta_t)**2

        # the solution v0 should be fixed throughout the batch
        if idx == self.config.N-1:
            xnow = tf.slice(x, [0,0,0], [1,1,self.config.dim])
            g0 = self.model(xnow, training, 'g', idx)
            g_now = tf.tile(g0, [self.config.batch_size, 1])
            g0 = self.eqn.symmetrise(g0)
            v0 = y0 + self.config.delta_t \
                        * self.eqn.f(xnow, y0, z0, g0)
        else:
            g_now = self.model(x[:, :-1, :], training, 'g', idx)

        g_now = self.eqn.symmetrise(g_now)

        loss += tf.reduce_mean((g_now-tf.stop_gradient(g_target)) ** 2)
        # we are done here for fully nonlinear model
        return loss, v0


########## Section configuration ####################
class Config:
    """Configurations for defining the problem and the solver."""
    def __init__(self, dim, T, N, dtype, batch_size, train_steps,
            lr_boundaries, lr_values, eqn_name, var_reduction,
            y_neurons, z_neurons, g_neurons, use_lstm):
        self.dim = dim
        self.T = T
        self.N = N
        self.dtype = dtype
        self.delta_t = self.T/self.N
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.lr_boundaries = lr_boundaries
        self.lr_values = lr_values
        self.eqn_name = eqn_name
        self.var_reduction = var_reduction
        self.y_neurons = y_neurons
        self.z_neurons = z_neurons
        self.g_neurons = g_neurons
        self.use_lstm  = use_lstm


########## Section main ####################
def simulate_close_form(eqn_name, T=0.1):

    N = int(T/.01)
    # float64 for a better precision, float32 for smaller memory
    dtype = tf.float32

    batch_size = 100000
    train_steps = 900
    lr_boundaries = [2*train_steps//3, 5*train_steps//6]
    lr_values = [0.1, 0.01, 0.001]

    expr_name = 'logs/'
    expr_name += eqn_name
    expr_name += f'_mc_T_{T}.csv'

    _file = open(expr_name, 'w')
    _file.write('d,T,N,run,y0,runtime\n')

    # not doing d > 1 for large N
    d_arrays = [1, 10, 100] if N < 100 else [1]
    for d in d_arrays:
        config = Config(d, T, N, dtype, batch_size, train_steps,
                        lr_boundaries, lr_values, eqn_name, False,
                        1, 1, 1, use_lstm=False)
        eqn = globals()[eqn_name](config)

        # 10 independent runs
        for run in range(10):
            # run on CPU to obtain reproducible results
            tf.random.set_seed(run)

            t_0 = time.time()
            x, _ = eqn.sde(N - 1)
            v0 = tf.reduce_mean(eqn.phi(x)).numpy() * np.exp(-eqn.r * T)
            t_1 = time.time()
            _file.write('%i, %f, %i, %i, %f, %f\n'
                        % (d, T, N, run, v0, t_1 - t_0))
            print(d, T, N, run, v0, t_1 - t_0)


def main(eqn_name, var_reduction, use_lstm=False, T=0.1):
    N = int(T/.01)
    # float64 for a better precision, float32 for smaller memory
    dtype = tf.float32

    batch_size = 256
    train_steps = 900
    lr_boundaries = [2*train_steps//3, 5*train_steps//6]
    lr_values = [0.1, 0.01, 0.001]

    # var_reduction = False

    expr_name = 'logs/'
    if not var_reduction: expr_name += 'no_'
    expr_name += 'var_reduction_'
    if use_lstm: expr_name += 'lstm_'
    expr_name += eqn_name
    expr_name += f'_T_{T}.csv'

    _file = open(expr_name, 'w')
    _file.write('d,T,N,run,y0,runtime\n')

    # not doing d > 1 for large N
    d_arrays = [1, 10, 100] if N < 50 else [1]
    for d in d_arrays:
        y_neurons = [d+10, d+10, 1]
        z_neurons = [d+10, d+10, d]
        g_neurons = [d+10, d+10, d*d]

        config = Config(d, T, N, dtype, batch_size, train_steps,
                    lr_boundaries, lr_values, eqn_name, var_reduction,
                    y_neurons, z_neurons, g_neurons, use_lstm=use_lstm)
        eqn = globals()[eqn_name](config)

        # 10 independent runs
        for run in range(10):
            # run on CPU to obtain reproducible results
            tf.random.set_seed(run)

            ppde_solver = PPDESolver(config, eqn)
            t_0 = time.time()
            ppde_solver.train()
            ppde_solver.plt_stats()
            t_1 = time.time()
            _file.write('%i, %f, %i, %i, %f, %f\n'
                        % (d, T, N, run, ppde_solver.v0, t_1 - t_0))
            print(d, T, N, run, ppde_solver.v0, t_1 - t_0)
            del ppde_solver
        del config, eqn
    _file.close()

if __name__ == '__main__':
    simulate_close_form('AsianOption', T=1)
    simulate_close_form('AsianOption')
    simulate_close_form('BarrierOption')
    # choice of ControlProblem, AsianOption, and BarrierOption
    # for other problems, add a new class under Section equation
    main('AsianOption',    var_reduction=False, use_lstm=True, T=1)
    main('AsianOption',    var_reduction=False, use_lstm=True)
    main('BarrierOption',  var_reduction=False, use_lstm=True)
    main('ControlProblem', var_reduction=True,  use_lstm=True)
    main('ControlProblem', var_reduction=False, use_lstm=True)
    main('AsianOption',    var_reduction=False, T=1)
    main('AsianOption',    var_reduction=False)
    main('BarrierOption',  var_reduction=False)
    main('ControlProblem', var_reduction=True)
    main('ControlProblem', var_reduction=False)
