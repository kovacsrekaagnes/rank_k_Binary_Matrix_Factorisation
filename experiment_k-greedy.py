from BMF import *
import numpy as np
import pandas as pd

test_set = ['zoo','heart', 'lymp', 'apb', 'tumor_w_missing','hepatitis_w_missing', 'audio_w_missing', 'votes_w_missing']
ks = [2, 5, 10]
n_rand = 70
greedy_types = ['mixed',
                'original', 'original-perturbed', 'original-transpose', 'original-perturbed-transpose',
                'revised', 'revised-perturbed', 'revised-transpose', 'revised-perturbed-transpose'] +\
               ['seed%s'% i for i in range(n_rand)]

df_index = pd.MultiIndex.from_product([ks, greedy_types])
df = pd.DataFrame(index=df_index, columns=test_set)
df_best = pd.DataFrame(index=ks, columns=test_set)

for name in test_set:
    X = np.loadtxt('./data/%s.txt' % name, dtype=float)
    X_out, row_weights, col_weights, idx_rows_reconstruct,\
    idx_cols_reconstruct, idx_zero_row, idx_zero_col, idx_rows_unique, idx_cols_unique = preprocess(X)

    for k in ks:
        error = X.size
        for greedy_type in greedy_types:

            mixed = False
            revised = False
            perturbed = False
            transpose = False
            r_seed = None

            if 'mixed' in greedy_type:
                mixed = True
            if 'revised' in greedy_type:
                revised = True
            if 'perturbed' in greedy_type:
                perturbed = True
            if 'transpose' in greedy_type:
                transpose = True
            for i in range(n_rand):
                if str(i) in greedy_type:
                    r_seed = i

            # without preprocessing
            A1, B1 = BMF_k_greedy_heur(X, k, None, None, mixed, revised, perturbed, transpose, r_seed)
            Z1 = boolean_matrix_product(A1, B1)
            error_1 = np.nansum(np.abs(X - Z1))

            # with preprocessing
            A2, B2 = BMF_k_greedy_heur(X_out, k, row_weights, col_weights, mixed, revised, perturbed, transpose, r_seed)
            A2, B2 = post_process_factorisation(A2, B2, idx_rows_reconstruct, idx_zero_row, idx_cols_reconstruct,
                                       idx_zero_col)
            Z2 = boolean_matrix_product(A2, B2)
            error_2 = np.nansum(np.abs(X - Z2))

            df.loc[(k, greedy_type), name] = error_1
            df.loc[(k, greedy_type + '-preprocessed'), name] = error_2

            if error_1 <= error_2:
                A = A1
                B = B1
                error_new = error_1
            else:
                A = A2
                B = B2
                error_new = error_2

            if error_new < error:
                error = error_new

                np.savetxt('./experiments/real_data/k-greedy/%s_%s_A.txt' % (name, k), A, fmt='%u')
                np.savetxt('./experiments/real_data/k-greedy/%s_%s_B.txt' % (name, k), B, fmt='%u')
                df_best.loc[k, name] = np.nansum(np.abs(X - boolean_matrix_product(A, B)))

        print('%s  done with %s' % (name, k))
        df_best.to_csv('./experiments/real_data/k-greedy.csv')



