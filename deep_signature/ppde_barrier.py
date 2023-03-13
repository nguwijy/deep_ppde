import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math
import time
from lib.bsde import FBSDE_BlackScholes as FBSDE
from lib.options import Barrier
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('../')


def sample_x0(batch_size, dim, device):
    sigma = 0.3
    mu = 0.08
    tau = 0.001
    z = torch.randn(batch_size, dim, device=device)
    x0 = torch.exp((mu-0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z) # lognormal
    return x0


def train(T,
        n_steps,
        d,
        mu,
        sigma,
        depth,
        rnn_hidden,
        ffn_hidden,
        max_updates,
        batch_size,
        lag,
        base_dir,
        device,
        method
        ):

    ts = torch.linspace(0,T,n_steps+1, device=device)
    barrier = Barrier()
    fbsde = FBSDE(d, mu, sigma, depth, rnn_hidden, ffn_hidden)
    fbsde.to(device)
    optimizer = torch.optim.RMSprop(fbsde.parameters(), lr=0.0005)

    losses = []
    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, d, device)
        if method=="bsde":
            loss, _, _ = fbsde.bsdeint(ts=ts, x0=x0, option=barrier, lag=lag)
        else:
            loss, _, _ = fbsde.conditional_expectation(ts=ts, x0=x0, option=barrier, lag=lag)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
        # testing
        if idx%10 == 0:
            with torch.no_grad():
                x0 = torch.ones(5000,d,device=device) # we do monte carlo
                loss, Y, payoff = fbsde.bsdeint(ts=ts,x0=x0,option=barrier,lag=lag)
                payoff = torch.exp(-mu*ts[-1])*payoff.mean()



    result = {"state":fbsde.state_dict(),
            "loss":losses}
    torch.save(result, os.path.join(base_dir, "result.pth.tar"))

    return Y[0,0,0].item()



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='./tmp/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--d', default=4, type=int)
    parser.add_argument('--max_updates', default=600, type=int)
    parser.add_argument('--ffn_hidden', default=[20,20], nargs="+", type=int)
    parser.add_argument('--rnn_hidden', default=20, type=int)
    parser.add_argument('--depth', default=3, type=int)
    parser.add_argument('--T', default=0.1, type=float)
    parser.add_argument('--n_steps', default=10, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--lag', default=10, type=int, help="lag in fine time discretisation to create coarse time discretisation")
    parser.add_argument('--mu', default=0.01, type=float, help="risk free rate")
    parser.add_argument('--sigma', default=0.1, type=float, help="risk free rate")
    parser.add_argument('--method', default="bsde", type=str, help="learning method", choices=["bsde","orthogonal"])

    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"

    results_path = os.path.join(args.base_dir, "BS", args.method)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    logfile = f"logs/signature_barrier_{args.T}.csv"
    with open(logfile, "w") as f:
        f.write('d,T,N,run,y0,runtime\n')

    for dd in [1, 10]:
        for run in range(10):

            np.random.seed(run)

            t_0 = time.time()
            y0 = train(T=args.T,
                n_steps=args.n_steps,
                # d=args.d,
                d=dd,
                mu=args.mu,
                sigma=args.sigma,
                depth=args.depth,
                # rnn_hidden=args.rnn_hidden,
                # ffn_hidden=args.ffn_hidden,
                rnn_hidden=dd+10,
                ffn_hidden=[dd+10,dd+10],
                max_updates=args.max_updates,
                batch_size=args.batch_size,
                lag=args.lag,
                base_dir=results_path,
                device=device,
                method=args.method)
            t_1 = time.time()

            print(dd, args.T, args.n_steps, run, y0, t_1 - t_0)
            with open(logfile, "a") as f:
                f.write('%i, %f, %i, %i, %f, %f\n'
                        % (dd, args.T, args.n_steps, run,
                            y0, t_1 - t_0))
