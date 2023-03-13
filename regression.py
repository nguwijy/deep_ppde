import os
import numpy as np
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def simulate(ppde_type, t, n, d, sde, phi, f, batch_size):
    x, feature, dw = sde(d, batch_size, n-1)

    for i in range(n):
        xnow, featurenow, dwnow = x[:, n-i-1, :], feature[:, n-i-1, :], dw[:, n-i-1, :]
        xnext, featurenext = x[:, n-i, :], feature[:, n-i, :]

        if i == n-2:
            # at t1, feature can be written as a function of xnow and hence,
            # the regression result suffers from multicollinearity in independet variable
            # to solve this we do regression only on xnow and xnow^2
            Xnow = np.concatenate( [np.ones((batch_size, 1)), xnow, xnow*xnow], axis=1 )
        else:
            Xnow = np.concatenate( [np.ones((batch_size, 1)), xnow, featurenow,
                        xnow*featurenow, xnow*xnow, featurenow*featurenow], axis=1 )

        if i == 0:
            # d x 1 dimensional Y
            Ytarget = phi(xnext, featurenext)
        else:
            if i == n-1:
                # at t1, feature can be written as a function of xnow and hence,
                # the regression result suffers from multicollinearity in independet variable
                # to solve this we do regression only on xnow and xnow^2
                Xnext = np.concatenate( [np.ones((batch_size, 1)), xnext, xnext*xnext], axis=1 )
            else:
                Xnext = np.concatenate( [np.ones((batch_size, 1)), xnext, featurenext,
                            xnext*featurenext, xnext*xnext, featurenext*featurenext], axis=1 )
            Ytemp = np.matmul( Xnext, Ycoeff )
            Ztemp = np.matmul( Xnext, Zcoeff )
            if ppde_type == 'fully-nonlinear' or ppde_type == 'fully-nonlinear_var_reduction':
                Gtemp = np.matmul( Xnext, Gcoeff )
                Ytarget = Ytemp + t / n * f(d, xnext, featurenext, Ytemp, Ztemp, Gtemp)
            else:
                # if ppde is semi-linear, we do not need the Gamma
                Ytarget = Ytemp + t / n * f(d, xnext, featurenext, Ytemp, Ztemp, Ztemp)

        if i == n-1:
            Ynow = np.mean(Ytarget, axis=0, keepdims=True)
        else:
            Ycoeff = doReg( Ytarget, Xnow )
            Ynow = np.matmul( Xnow, Ycoeff )

        # here Ytarget is (1,) but sig_inverse is (d, d)
        # hence expand_dims is needed for tensor multiplication
        sig_inverse = sigma_inverse(d, batch_size, xnow)
        if ppde_type == 'semilinear_var_reduction' or ppde_type == 'fully-nonlinear_var_reduction':
            # for variance reduction, we minus Ynow
            Ztarget = np.expand_dims(Ytarget - Ynow, -1) \
                        * np.matmul(sig_inverse.transpose(0,2,1), np.expand_dims(dwnow, -1)) / (t/n)
        else:
            Ztarget = np.expand_dims(Ytarget, -1) \
                        * np.matmul(sig_inverse.transpose(0,2,1), np.expand_dims(dwnow, -1)) / (t/n)

        if i == n-1:
            xnow = np.mean(xnow, axis=0, keepdims=True)
            featurenow = np.mean(featurenow, axis=0, keepdims=True)
            Znow = np.mean(np.squeeze(Ztarget, -1), axis=0, keepdims=True)
        else:
            Zcoeff =  doReg( np.squeeze(Ztarget, -1), Xnow )
            Znow = np.matmul( Xnow, Zcoeff )

        if ppde_type == 'fully-nonlinear_var_reduction':
            dw2 = np.matmul(np.expand_dims(dwnow, -1), np.expand_dims(dwnow, -1).transpose(0,2,1))
            Gtarget = ( np.expand_dims(Ytarget - Ynow, -1) \
                            - np.matmul( np.matmul( sigma(d, batch_size, Xnow).transpose(0,2,1),
                                        np.expand_dims(Znow, -1) ).transpose(0,2,1),
                                    np.expand_dims(dwnow, -1) ) ) \
                            * np.matmul( np.matmul( sig_inverse.transpose(0,2,1),
                                (dw2 - t/n * np.stack([np.eye(d)]*batch_size, axis=0))), sig_inverse ) \
                            / (t/n)**2
            Gtarget = np.reshape( Gtarget, (batch_size, d*d) )

            if i == n-1:
                Gnow = np.mean(Gtarget, axis=0, keepdims=True)
                return Ynow + t / n * f(d, xnow, featurenow, Ynow, Znow, Gnow)
            else:
                Gcoeff =  doReg( Gtarget, Xnow )

        if ppde_type == 'fully-nonlinear':
            dw2 = np.matmul(np.expand_dims(dwnow, -1), np.expand_dims(dwnow, -1).transpose(0,2,1))
            Gtarget = np.expand_dims(Ytarget, -1) \
                            * np.matmul( np.matmul( sig_inverse.transpose(0,2,1),
                                (dw2 - t/n * np.stack([np.eye(d)]*batch_size, axis=0))), sig_inverse ) \
                            / (t/n)**2
            Gtarget = np.reshape( Gtarget, (batch_size, d*d) )

            if i == n-1:
                Gnow = np.mean(Gtarget, axis=0, keepdims=True)
                return Ynow + t / n * f(d, xnow, featurenow, Ynow, Znow, Gnow)
            else:
                Gcoeff =  doReg( Gtarget, Xnow )

    return Ynow + t / n * f(d, xnow, featurenow, Ynow, Znow, Znow)


def doReg(Ymat, Xmat):
    XXinv = np.linalg.inv( np.matmul(Xmat.T, Xmat) )
    XY = np.matmul(Xmat.T, Ymat)
    return np.matmul( XXinv, XY )


def phi(x, feature):
    if which_type == 'xiaolu_control':
        x = np.mean(x, axis=1)
        feature = np.mean(feature, axis=1)
        return np.expand_dims( np.cos(x + feature), -1 )
    elif which_type == 'xiaolu_asian':
        K = 0.7
        # average along the dimension axis
        feature = np.mean(feature, axis=1, keepdims=True)
        return np.where( feature-K>0, feature-K, np.zeros_like(feature) )
    elif which_type == 'xiaolu_barrier':
        K, B = 0.7, 1.2
        # average along the dimension axis
        x = np.mean(x, axis=1, keepdims=True)
        up_flag = B - feature
        return np.where(np.logical_and(up_flag > 0, x-K > 0), x-K, np.zeros_like(x))


def f(_d, x, feature, y, z, gamma):
    if which_type == 'xiaolu_control':
        mu_low, mu_high, sig_low, sig_high = -0.2, 0.2, 0.2, 0.3
        a_low, a_high = sig_low**2, sig_high**2
        x = np.mean(x, axis=1)
        feature = np.mean(feature, axis=1)
        reduced_sin = np.sin(x + feature)
        reduced_cos = np.cos(x + feature)
        reduced_z = np.sum(z, axis=1)
        # because gamma before transformation has the shape
        # (batch_size, dxd) instead of (batch_size, d, d)
        gamma = np.reshape( gamma, (x.shape[0], d, d) )
        reduced_gamma = np.trace( gamma, axis1=1, axis2=2 )
        cancel = -a_low * reduced_gamma / 2
        mu = np.where(reduced_z>0, mu_low*reduced_z, mu_high*reduced_z)
        a = np.where(reduced_gamma>0, a_high*reduced_gamma, a_low*reduced_gamma) / 2
        small_f = np.where(reduced_sin>0, x + mu_high, x + mu_low) \
                    * reduced_sin + 1/_d * np.where(reduced_cos>0, a_low*reduced_cos/2 , a_high*reduced_cos/2)
        return np.expand_dims( cancel+mu+a+small_f, -1 )
    elif which_type == 'xiaolu_barrier' or which_type == 'xiaolu_asian':
        r = 0.01
        return - r*y


def b(_d, batch_size, x):
    if which_type == 'xiaolu_control':
        return np.zeros_like(x)
    elif which_type == 'xiaolu_barrier' or which_type == 'xiaolu_asian':
        r = 0.01
        return r*x


def sigma_inverse(_d, batch_size, x):
    if which_type == 'xiaolu_control':
        sig_low = 0.2
        # np.eye creates identity matrix
        return 1 / sig_low * np.stack([np.eye(_d)]*batch_size, axis=0)

    elif which_type == 'xiaolu_barrier' or which_type == 'xiaolu_asian':
        sig = 0.1
        # return 1 / sig * np.diag(np.reciprocal(x))

        return 1 / sig * np.stack( [np.diag(np.reciprocal(x[i])) for i in range(batch_size)], axis=0 )
    else:
        # for a general sigma matrix which may not be diagonal
        # we can only use inv function in np.linalg
        return np.linalg.inv(sigma(_d, batch_size, x))


def sigma(_d, batch_size, x):
    if which_type == 'xiaolu_control':
        sig_low = 0.2
        # np.eye creates identity matrix
        return sig_low * np.stack([np.eye(_d)]*batch_size, axis=0)
    elif which_type == 'xiaolu_barrier' or which_type == 'xiaolu_asian':
        sig = 0.1
        # return sig * np.diag(x)
        return sig * np.stack( [np.diag(x[i]) for i in range(batch_size)], axis=0 )


def init_x():
    if which_type == 'xiaolu_control':
        return 0
    elif which_type == 'xiaolu_barrier' or which_type == 'xiaolu_asian':
        return 1


def sde(_d, batch_size, n):
    temptemp = init_x()*np.ones((batch_size, _d))
    x = [ temptemp ]
    dw = [ ]
    if which_type == 'xiaolu_asian' or which_type == 'xiaolu_control':
        feature = [ np.zeros((batch_size, _d)) ]
    elif which_type == 'xiaolu_barrier':
        feature = [ np.mean(temptemp, axis=1, keepdims=True) ]

    for _n in range(n + 1):
        dwnow = np.random.normal(0, np.sqrt(T / N), (batch_size, _d))
        # squeeze used because matmul produces dim of d*1 while the rest are d
        temp =  x[-1] + b(_d, batch_size, x[-1]) * T / N \
                + np.squeeze(np.matmul(sigma(_d, batch_size, x[-1]), np.expand_dims(dwnow, -1)), -1)
        dw.append(dwnow)
        x.append(temp)
        if which_type == 'xiaolu_asian':
            feature.append( (feature[-1]*_n*(T/N)+(temptemp + temp)*0.5*T/N) / ((_n+1)*(T/N)) )
        elif which_type == 'xiaolu_control':
            feature.append( feature[-1]+(temptemp + temp)*0.5*T/N )
        elif which_type == 'xiaolu_barrier':
            feature.append( np.maximum(feature[-1], np.mean(temp, axis=1, keepdims=True)) )
        temptemp = temp
    dw = np.stack(dw, axis=1)
    x = np.stack(x, axis=1)
    feature = np.stack(feature, axis=1)
    return x, feature, dw


# -------------------- main script starts heres --------------------------------
batch_size = 10000

T = 0.1
for which_type in [ 'xiaolu_asian', 'xiaolu_barrier', 'xiaolu_control' ]:
    _file = open(f'logs/{which_type}_T_{T}.csv', 'w')
    _file.write('d,T,N,run,y0,runtime\n')
    print(which_type)

    for d in [1, 10, 100]:
        for N in [10]:
            for run in range(10):
                t_0 = time.time()
                if which_type == 'xiaolu_control':
                    y0 = simulate('fully-nonlinear_var_reduction', T, N, d, sde, phi, f, batch_size)
                elif which_type == 'xiaolu_barrier' or which_type == 'xiaolu_asian':
                    y0 = simulate('semilinear_var_reduction', T, N, d, sde, phi, f, batch_size)
                t_1 = time.time()

                _file.write('%i, %f, %i, %i, %f, %f\n'
                            % (d, T, N, run, y0[0,0], t_1 - t_0))
                _file.flush()
                print(d, T, N, run, y0[0,0], t_1 - t_0)

    _file.close()

for which_type in [ 'xiaolu_asian', 'xiaolu_barrier', 'xiaolu_control' ]:
    _file = open(f'logs/no_var_reduction_{which_type}_T_{T}.csv', 'w')
    _file.write('d,T,N,run,y0,runtime\n')
    print(which_type)

    for d in [1, 10, 100]:
        for N in [10]:
            for run in range(10):
                t_0 = time.time()
                if which_type == 'xiaolu_control':
                    y0 = simulate('fully-nonlinear', T, N, d, sde, phi, f, batch_size)
                elif which_type == 'xiaolu_barrier' or which_type == 'xiaolu_asian':
                    y0 = simulate('semilinear', T, N, d, sde, phi, f, batch_size)
                t_1 = time.time()

                _file.write('%i, %f, %i, %i, %f, %f\n'
                            % (d, T, N, run, y0[0,0], t_1 - t_0))
                _file.flush()
                print(d, T, N, run, y0[0,0], t_1 - t_0)

    _file.close()
