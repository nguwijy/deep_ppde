import os
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.chdir('./logs')
file_log_path = 'lit_compare.csv'

exact_sol_asian = {}
df = pd.read_csv("AsianOption_mc_T_0.1.csv")
for dd in [1, 10, 100]:
    tmp = df.loc[df['d'] == dd]
    exact_sol_asian[dd] = tmp['y0'].mean()

exact_sol_asian_T_1 = {}
df = pd.read_csv("AsianOption_mc_T_1.csv")
for dd in [1]:
    tmp = df.loc[df['d'] == dd]
    exact_sol_asian_T_1[dd] = tmp['y0'].mean()

exact_sol_barrier = {}
df = pd.read_csv("BarrierOption_mc_T_0.1.csv")
for dd in [1, 10, 100]:
    tmp = df.loc[df['d'] == dd]
    exact_sol_barrier[dd] = tmp['y0'].mean()

exact_sol_control = 1


control_csv = [ 'no_var_reduction_ControlProblem_T_0.1.csv',
                'var_reduction_ControlProblem_T_0.1.csv',
                'var_reduction_lstm_ControlProblem_T_0.1.csv',
                'no_var_reduction_xiaolu_control_T_0.1.csv', 'xiaolu_control_T_0.1.csv',
                'galerkin_control_T_0.1.csv' ]
asian_csv = [ 'no_var_reduction_AsianOption_T_0.1.csv',
              'no_var_reduction_AsianOption_T_0.1.csv',  # not used
              'no_var_reduction_lstm_AsianOption_T_0.1.csv',
              'xiaolu_asian_T_0.1.csv', 'xiaolu_asian_T_0.1.csv',
              'galerkin_asian_T_0.1.csv', 'signature_asian_T_0.1.csv',
              'weinan_asian_T_0.1.csv', 'ariel_asian_T_0.1.csv' ]
asian_csv_T_1 = [ 'no_var_reduction_AsianOption_T_1.csv',
                'no_var_reduction_AsianOption_T_1.csv',  # not used
                'no_var_reduction_lstm_AsianOption_T_1.csv',
                'xiaolu_asian_T_1.csv', 'xiaolu_asian_T_1.csv',
                'galerkin_asian_T_1.csv', 'signature_asian_T_1.csv',
                'weinan_asian_T_1.csv', 'ariel_asian_T_1.csv' ]
barrier_csv = [ 'no_var_reduction_BarrierOption_T_0.1.csv',
                'no_var_reduction_BarrierOption_T_0.1.csv',  # not used
                'no_var_reduction_lstm_BarrierOption_T_0.1.csv',
                'xiaolu_barrier_T_0.1.csv', 'xiaolu_barrier_T_0.1.csv',
                'galerkin_barrier_T_0.1.csv', 'signature_barrier_T_0.1.csv' ]

scheme_name = [ 'Deep PPDE using \eqref{eq:deep_scheme_nonlinear}',
                'Deep PPDE using \eqref{eq:var_reduced_deep_scheme_nonlinear}',
                'Deep PPDE with LSTM',
                '\cite{ren2017convergence} using \eqref{eq:probabilistic_scheme}',
                '\cite{ren2017convergence} using \eqref{eq:var_reduced_probabilistic_scheme}',
                '\cite{saporito2020pdgm}',
                '\cite{sabate2020solving}',
                '\cite{han2018solving}',
                '\cite{beck2019deep}' ]
deep_or_not = [ 'Y', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'Y', 'Y' ]

res = []
# first line, use write instead of append
with open(file_log_path, 'w') as f:
    f.write('\hline\n')
for ii in range(len(control_csv)):
    csv = control_csv[ii]
    name = scheme_name[ii]
    deep = deep_or_not[ii]
    for dd in [1, 10, 100]:
        # saporito and zhang, sabate et al do not have result for 100-dimensional
        if dd==100 and (ii==5 or ii==6):   continue

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
            res.append([
                'Control', name, deep, dd, L1, sd, exact_sol_control, error_L1, int(round(avg_runtime))
            ])
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
        if dd==100 and (ii==5 or ii==6):   continue
        # weinan and ariel only have result for 1-dimensional
        if (dd==100 or dd==10) and (ii==7 or ii==8):   continue

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
            res.append([
                'Asian, T=.1', name, deep, dd, L1, sd, exact_sol_asian[dd], error_L1, int(round(avg_runtime))
            ])
            f.write('{:s} & {:s} & {:d} & {:.7g} & {:.2E} & {:.7g} & {:.2E} & {:d} \\\\\n'
                    .format(name, deep, dd, L1, sd, exact_sol_asian[dd], error_L1,
                            int(round(avg_runtime))))
    with open(file_log_path, 'a') as f:
        f.write('\hline\n')

with open(file_log_path, 'a') as f:
    f.write('\n\n\n\hline\n')

for ii in range(len(asian_csv_T_1)):
    csv = asian_csv_T_1[ii]
    name = scheme_name[ii]
    deep = deep_or_not[ii]
    for dd in [1]:
        df = pd.read_csv(csv)
        df = df.loc[df['d'] == dd]
        relative_error = abs( df['y0'] - exact_sol_asian_T_1[dd])/exact_sol_asian_T_1[dd]
        df['relative_error'] = relative_error
        L1 = df['y0'].mean()
        sd = df['y0'].std()
        error_L1 = df['relative_error'].mean()
        error_sd = df['relative_error'].std()
        avg_runtime = df['runtime'].mean()
        with open(file_log_path, 'a') as f:
            res.append([
                'Asian, T=1', name, deep, dd, L1, sd, exact_sol_asian_T_1[dd], error_L1, int(round(avg_runtime))
            ])
            f.write('{:s} & {:s} & {:d} & {:.7g} & {:.2E} & {:.7g} & {:.2E} & {:d} \\\\\n'
                    .format(name, deep, dd, L1, sd, exact_sol_asian_T_1[dd], error_L1,
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
        if dd==100 and (ii==5 or ii==6):   continue

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
            res.append([
                'Barrier', name, deep, dd, L1, sd, exact_sol_barrier[dd], error_L1, int(round(avg_runtime))
            ])
            f.write('{:s} & {:s} & {:d} & {:.7g} & {:.2E} & {:.7g} & {:.2E} & {:d} \\\\\n'
                    .format(name, deep, dd, L1, sd, exact_sol_barrier[dd], error_L1,
                    int(round(avg_runtime))))
    with open(file_log_path, 'a') as f:
        f.write('\hline\n')

res = pd.DataFrame(res, columns=[
    'problem_name', 'scheme_name', 'deep', 'dim', 'mean', 'std', 'exact', 'l1_error', 'runtime'
])
