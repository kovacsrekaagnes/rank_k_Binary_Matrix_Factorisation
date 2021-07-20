from BMF import *
import time
import pandas as pd
import numpy as np

test_set = ['zoo', 'heart', 'lymp', 'apb','tumor_w_missing','hepatitis_w_missing', 'audio_w_missing', 'votes_w_missing']

ks = [2, 5, 10]

MLP_max_time = 1200
MIP_max_time = 600

k_index = ['k=%s' %k for k in ks]
MLP_name_index = ['MLP(1)_obj','MLP(1)_dual','MLP(1)_#cols','MLP(1)_time']
MIP_name_index = ['MIP(1)_obj', 'MIP(1)_gap','MIP(1)_time', 'Error']

MLP_col_index = pd.MultiIndex.from_product([k_index, MLP_name_index])
MIP_col_index = pd.MultiIndex.from_product([k_index, MIP_name_index])

row_index = test_set

MLP_df = pd.DataFrame(index=MLP_col_index, columns=row_index)
MIP_df = pd.DataFrame(index=MIP_col_index, columns=row_index)


for name in test_set:

    X = np.loadtxt('./data/%s.txt' % name, dtype=float)

    for k in ks:
        A_greedy = np.loadtxt('./k-greedy/%s_%s_A.txt' % (name, k), dtype=int)
        B_greedy = np.loadtxt('./k-greedy/%s_%s_B.txt' % (name, k), dtype=int)

        bmf = BMF_via_CG(X, k, A_init=A_greedy, B_init=B_greedy)
        bmf.preprocess_input_matrices()

        start = time.time()
        bmf.MLP_solve(max_time=MLP_max_time, display=False, log=True)
        end = time.time() - start

        bmf.post_process_output_matrices()
        np.savetxt('./experiments/real_data/MLP/%s_k%s_A_MLP1_time%s.txt' % (name, k, MLP_max_time), bmf.MLP_A, fmt='%u')
        np.savetxt('./experiments/real_data/MLP/%s_k%s_B_MLP1_time%s.txt' % (name, k, MLP_max_time), bmf.MLP_B, fmt='%u')
        bmf.preprocess_input_matrices()

        MLP_df.loc[('k=%s' % k, 'MLP(1)_obj'), name] = bmf.MLP_model.solution.get_objective_value()
        MLP_df.loc[('k=%s' % k, 'MLP(1)_#cols'), name] = bmf.MLP_A.shape[1]
        MLP_df.loc[('k=%s' % k, 'MLP(1)_dual'), name] = bmf.best_dual_bound
        MLP_df.loc[('k=%s' % k, 'MLP(1)_time'), name] = end

        MLP_df.to_csv('./experiments/real_data/MLP1_time_%s_zoo.csv' % (MLP_max_time))
        bmf.MLP_log.to_csv('./experiments/real_data/MLP/%s_k%s_MLP1_time%s.csv' % (name, k, MLP_max_time))

        print(MLP_df)
        print('\n\n\n Done with MLP on %s with k=%s \n\n\n' % (name, k))

        start = time.time()
        bmf.MIP_solve(max_time=MIP_max_time)
        end = time.time() - start
        bmf.post_process_output_matrices()

        np.savetxt('./experiments/real_data/MIP/%s_k%s_A_MLP1_time%s_MIP_time%s.txt' % (name, k, MLP_max_time, MIP_max_time), bmf.A, fmt='%u')
        np.savetxt('./experiments/real_data/MIP/%s_k%s_B_MLP1_time%s_MIP_time%s.txt' % (name, k, MLP_max_time, MIP_max_time), bmf.A, fmt='%u')

        MIP_df.loc[('k=%s' % k, 'MIP(1)_obj'), name] = bmf.MIP_model.solution.get_objective_value()
        MIP_df.loc[('k=%s' % k, 'MIP(1)_gap'), name] = 100 * \
                                                      (bmf.MIP_model.solution.get_objective_value() -
                                                       bmf.best_dual_bound) /\
                                                      (1.0e-10 + bmf.MIP_model.solution.get_objective_value())
        MIP_df.loc[('k=%s' % k, 'MIP(1)_time'), name] = end
        MIP_df.loc[('k=%s' % k, 'Error'), name] = np.nansum(np.abs(X - boolean_matrix_product(bmf.A, bmf.B)))

        print('\n\n\n Done with MIP on %s with k=%s \n\n\n' % (name, k))

        print(MIP_df)

        MIP_df.to_csv('./experiments/real_data/MLP1_time%s_MIP_time%s.csv' % (MLP_max_time, MIP_max_time))






