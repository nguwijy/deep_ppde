import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('./logs')
file_log_path = 'lit_compare.csv'

exact_sol_asian = {}
df = pd.read_csv("mc_asian.csv")
for dd in [1, 10, 100]:
    tmp = df.loc[df['d'] == dd]
    exact_sol_asian[dd] = tmp['y0'].mean()

exact_sol_barrier = {}
df = pd.read_csv("mc_barrier.csv")
for dd in [1, 10, 100]:
    tmp = df.loc[df['d'] == dd]
    exact_sol_barrier[dd] = tmp['y0'].mean()

exact_sol_control = 1


control_csv = [ 'no_var_reduction_ControlProblem.csv',
                'var_reduction_ControlProblem.csv',
                'no_var_reduction_xiaolu_control.csv', 'xiaolu_control.csv',
                'galerkin_control.csv' ]
asian_csv = [ 'no_var_reduction_AsianOption.csv',
                'no_var_reduction_AsianOption.csv',  # not used
                'no_var_reduction_xiaolu_asian.csv', 'xiaolu_asian.csv',
                'galerkin_asian.csv', 'signature_asian.csv',
                'weinan_asian.csv', 'ariel_asian.csv' ]
barrier_csv = [ 'no_var_reduction_BarrierOption.csv',
                'no_var_reduction_BarrierOption.csv',  # not used
                'no_var_reduction_xiaolu_barrier.csv', 'xiaolu_barrier.csv',
                'galerkin_barrier.csv', 'signature_barrier.csv' ]

scheme_name = [ 'Deep PPDE using \eqref{eq:deep_scheme_nonlinear}',
                'Deep PPDE using \eqref{eq:var_reduced_deep_scheme_nonlinear}',
                '\cite{ren2017convergence} using \eqref{eq:probabilistic_scheme}',
                '\cite{ren2017convergence} using \eqref{eq:var_reduced_probabilistic_scheme}',
                '\cite{saporito2020pdgm}',
                '\cite{sabate2020solving}',
                '\cite{han2018solving}',
                '\cite{beck2019deep}' ]
deep_or_not = [ 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'Y', 'Y' ]

# first line, use write instead of append
with open(file_log_path, 'w') as f:
    f.write('\hline\n')
for ii in range(len(control_csv)):
    csv = control_csv[ii]
    name = scheme_name[ii]
    deep = deep_or_not[ii]
    for dd in [1, 10, 100]:
        # saporito and zhang, sabate et al do not have result for 100-dimensional
        if dd==100 and (ii==4 or ii==5):   continue

        df = pd.read_csv(csv)
        df = df.loc[df['d'] == dd]
        relative_error = abs( df['y0'] - exact_sol_control)/exact_sol_control
        df['relative_error'] = relative_error
        L1 = df['y0'].mean()
        sd = df['y0'].std()
        error_L1 = df['relative_error'].mean()
        # error_sd = df['relative_error'].std()
        avg_runtime = df['runtime'].mean()
        with open(file_log_path, 'a') as f:
            f.write('{:s} & {:s} & {:d} & {:.7g} & {:.2E} & {:.7g} & {:.2E} & {:d} \\\\\n'
                    .format(name, deep, dd, L1, sd, exact_sol_control, error_L1,
                    int(round(avg_runtime))))
    with open(file_log_path, 'a') as f:
        f.write('\hline\n')

with open(file_log_path, 'a') as f:
    f.write('\n\n\n\hline\n')

for ii in range(len(asian_csv)):
    csv = asian_csv[ii]
    name = scheme_name[ii]
    deep = deep_or_not[ii]
    for dd in [1, 10, 100]:
        # saporito and zhang, sabate et al do not have result for 100-dimensional
        if dd==100 and (ii==4 or ii==5):   continue
        # weinan and ariel only have result for 1-dimensional
        if (dd==100 or dd==10) and (ii==6 or ii==7):   continue

        df = pd.read_csv(csv)
        df = df.loc[df['d'] == dd]
        relative_error = abs( df['y0'] - exact_sol_asian[dd])/exact_sol_asian[dd]
        df['relative_error'] = relative_error
        L1 = df['y0'].mean()
        sd = df['y0'].std()
        error_L1 = df['relative_error'].mean()
        error_sd = df['relative_error'].std()
        avg_runtime = df['runtime'].mean()
        with open(file_log_path, 'a') as f:
            f.write('{:s} & {:s} & {:d} & {:.7g} & {:.2E} & {:.7g} & {:.2E} & {:d} \\\\\n'
                    .format(name, deep, dd, L1, sd, exact_sol_asian[dd], error_L1,
                    int(round(avg_runtime))))
    with open(file_log_path, 'a') as f:
        f.write('\hline\n')

with open(file_log_path, 'a') as f:
    f.write('\n\n\n\hline\n')

for ii in range(len(barrier_csv)):
    csv = barrier_csv[ii]
    name = scheme_name[ii]
    deep = deep_or_not[ii]
    for dd in [1, 10, 100]:
        # saporito and zhang, sabate et al do not have result for 100-dimensional
        if dd==100 and (ii==4 or ii==5):   continue

        df = pd.read_csv(csv)
        df = df.loc[df['d'] == dd]
        relative_error = abs( df['y0'] - exact_sol_barrier[dd])/exact_sol_barrier[dd]
        df['relative_error'] = relative_error
        L1 = df['y0'].mean()
        sd = df['y0'].std()
        error_L1 = df['relative_error'].mean()
        error_sd = df['relative_error'].std()
        avg_runtime = df['runtime'].mean()
        with open(file_log_path, 'a') as f:
            f.write('{:s} & {:s} & {:d} & {:.7g} & {:.2E} & {:.7g} & {:.2E} & {:d} \\\\\n'
                    .format(name, deep, dd, L1, sd, exact_sol_barrier[dd], error_L1,
                    int(round(avg_runtime))))
    with open(file_log_path, 'a') as f:
        f.write('\hline\n')
