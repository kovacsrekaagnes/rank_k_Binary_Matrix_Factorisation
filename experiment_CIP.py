from BMF import *
import time
import pandas as pd
import numpy as np

test_set = ['zoo','heart', 'lymp','apb', 'tumor_w_missing','hepatitis_w_missing', 'audio_w_missing', 'votes_w_missing']

ks = [2, 5, 10]

max_time = 1200

k_index = ['k=%s' %k for k in ks]
name_index = ['CIP_obj','CIP_dual','CIP_time', 'CIP_gap', 'CIP_error']

col_index = pd.MultiIndex.from_product([k_index, name_index])
row_index = test_set

df = pd.DataFrame(index=col_index, columns=row_index)

for name in test_set:

    X = np.loadtxt('./data/%s.txt' % name, dtype=float)

    for k in ks:
        A_greedy = np.loadtxt('./k-greedy/%s_%s_A.txt' % (name, k), dtype=int)
        B_greedy = np.loadtxt('./k-greedy/%s_%s_B.txt' % (name, k), dtype=int)

        bmf = BMF_via_compact_IP(X, k, A_init=A_greedy, B_init=B_greedy)
        bmf.preprocess_input_matrices()

        start = time.time()
        bmf.CIP_solve(max_time=max_time)
        end = time.time() - start

        bmf.post_process_output_matrices()

        np.savetxt('./experiments/real_data/CIP/%s_k%s_A_MLP1_time%s.txt' % (name, k, max_time), bmf.A, fmt='%u')
        np.savetxt('./experiments/real_data/CIP/%s_k%s_B_MLP1_time%s.txt' % (name, k, max_time), bmf.B, fmt='%u')

        df.loc[('k=%s' % k, 'CIP_obj'), name] = bmf.CIP_model.solution.get_objective_value()
        df.loc[('k=%s' % k, 'CIP_dual'), name] = bmf.CIP_model.solution.MIP.get_best_objective()
        df.loc[('k=%s' % k, 'CIP_time'), name] = end
        df.loc[('k=%s' % k, 'CIP_gap'), name] = bmf.CIP_model.solution.MIP.get_mip_relative_gap() * 100
        df.loc[('k=%s' % k, 'CIP_error'), name] = np.nansum(np.abs(X - boolean_matrix_product(bmf.A, bmf.B)))

        print('\n\n\n Done with CIP on %s with k=%s \n\n\n' % (name, k))

        print(df)

        df.to_csv('./experiments/real_data/CIP_time%s.csv' % (max_time))






