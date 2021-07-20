from BMF import boolean_matrix_product
from sklearn.decomposition import non_negative_factorization
import numpy as np
import pandas as pd
import time
EPS = np.finfo(np.float32).eps

test_set = ['zoo', 'heart','lymp','apb']#,'tumor_w_missing','hepatitis_w_missing', 'audio_w_missing', 'votes_w_missing']
ks = [2, 5, 10]
df_index = pd.MultiIndex.from_product([['error', 'time'], ['k=%s' %k for k in ks]])
df = pd.DataFrame(index=df_index, columns=test_set)

for name in test_set:
    X = np.loadtxt('./data/%s.txt' % name, dtype=float)
    (n,m) = X.shape
    for k in ks:

        start = time.time()
        out = non_negative_factorization(X, n_components=k)
        end = time.time()

        A_NMF = out[0]
        B_NMF = out[1]

        A = np.zeros([n, k], dtype=int)
        A[A_NMF > 0.5] = 1
        B = np.zeros([k, m], dtype=int)
        B[B_NMF > 0.5] = 1

        t = end - start
        error = np.sum(np.abs(X - boolean_matrix_product(A, B)))

        df.loc[('error', 'k=%s' % k), name] = error
        df.loc[('time', 'k=%s' % k), name] = t

        df.to_csv('./experiments/real_data/nmf.csv')

        print('\n\nDone with %s k=%s \n\n' %(name,k))

