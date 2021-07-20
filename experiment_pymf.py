import numpy as np
from BMF import boolean_matrix_product
from pymf import BNMF
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

        n_iter = 10000

        start = time.time()

        handle = BNMF(X, num_bases=k)
        handle.factorize(niter=n_iter)
        B_chris = handle.H
        A_chris = handle.W

        end = time.time()

        A = np.zeros([n, k], dtype=int)
        A[A_chris > 1. - EPS ] = 1
        B = np.zeros([k, m], dtype=int)
        B[B_chris > 1. - EPS] = 1

        Z = boolean_matrix_product(A, B)

        t = end - start
        error = np.sum(np.abs(X - Z))

        df.loc[('error', 'k=%s' % k), name] = error
        df.loc[('time', 'k=%s' % k), name] = t

        df.to_csv('./experiments/real_data/pymf.csv')

        print('\n\nDone with %s k=%s \n\n' %(name,k))


