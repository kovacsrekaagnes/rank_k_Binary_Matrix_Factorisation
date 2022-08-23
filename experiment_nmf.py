from BMF import boolean_matrix_product
from sklearn.decomposition import non_negative_factorization
import numpy as np
import pandas as pd
import time
EPS = np.finfo(np.float32).eps

#convert NMF factorisation to BMF, and first scale NMF factors
def from_NMF_to_BMF(A_NMF,B_NMF, threshold=0.5):

    k = A_NMF.shape[1]

    #scale NMF factors
    for p in range(k):
        max_Ap = A_NMF[:, p].max()
        max_Bp = B_NMF[p, :].max()
        A_NMF[:, p] = A_NMF[:, p] / np.sqrt(max_Ap) * np.sqrt(max_Bp)
        B_NMF[p, :] = B_NMF[p, :] / np.sqrt(max_Bp) * np.sqrt(max_Ap)

    #threshold NMF to get BMF
    A_BMF = np.zeros(A_NMF.shape, dtype=int)
    A_BMF[A_NMF > threshold] = 1
    B_BMF = np.zeros(B_NMF.shape, dtype=int)
    B_BMF[B_NMF > threshold] = 1

    return A_BMF, B_BMF

test_set = ['zoo', 'heart','lymp','apb']#,'tumor_w_missing','hepatitis_w_missing', 'audio_w_missing', 'votes_w_missing']
ks = [2, 5, 10]
df_index = pd.MultiIndex.from_product([['error', 'time', 'threshold', 'init'], ['k=%s' %k for k in ks]])
df = pd.DataFrame(index=df_index, columns=test_set)

for name in test_set:
    X = np.loadtxt('./data/%s.txt' % name, dtype=float)
    (n,m) = X.shape
    for k in ks:

        #initialise a large error value, just to be able to compare later and record best
        df.loc[('error', 'k=%s' % k), name] = n*m

        #try all possible NMF initalisations
        for init_NMF in ('nndsvda', 'nndsvd', 'nndsvdar', 'random'):

            start = time.time()
            A_NMF, B_NMF, _ = non_negative_factorization(X, n_components=k, init=init_NMF)
            end = time.time()
            t = end - start

            #try a lot of thresholds to get BMF
            for threshold in np.arange(0.1, 1, 0.1):
                start = time.time()
                A, B = from_NMF_to_BMF(A_NMF, B_NMF, threshold)
                end = time.time()

                error = np.sum(np.abs(X - boolean_matrix_product(A, B)))

                #only save if current BMF is better than previous best
                if error < df.loc[('error', 'k=%s' % k), name]:

                    df.loc[('error', 'k=%s' % k), name] = error
                    df.loc[('time', 'k=%s' % k), name] = t + end - start
                    df.loc[('threshold', 'k=%s' % k), name] = threshold
                    df.loc[('init', 'k=%s' % k), name] = init_NMF

        df.to_csv('./experiments/real_data/nmf.csv')

        print('\n\nDone with %s k=%s \n\n' %(name,k))

