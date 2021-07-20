'''
this work implements the rank-k binary matrix factorisation integer programs in

1.) R. A. Kovacs, O. Gunluk, R. A. Hauser, Binary Matrix Factorisation and Completion via Integer Programming (2021)
    URL https://arxiv.org/abs/2106.13434

2.) R. A. Kovacs, O. Gunluk, R. A. Hauser, Binary matrix factorisation via column generation,
    Proceedings of the AAAI Conference on Artificial Intelligence 35 (5) (2021) 3823–3831.
    URL https://ojs.aaai.org/index.php/AAAI/article/view/16500  &   https://arxiv.org/abs/2011.04457

3.) R. A. Kovacs, O. Gunluk, R. A. Hauser, Low-rank boolean matrix approximation by integer programming,
    NIPS, Optimization for Machine Learning Workshop, 2017
    URL https://opt-ml.org/papers/OPT2017_paper_34.pdf  &   https://arxiv.org/abs/1803.04825
'''

import numpy as np
import pandas as pd
import cplex
from math import ceil

# float comparison
EPS = np.finfo(np.float32).eps
# optimality tolerances... in some cases 1.0e-6 resulted in numerical precision errors, and infinite iterations for CG
TOL = 1.0e-4

def boolean_matrix_product(A, B):
    """
    Compute the Boolean matrix product of A and B

    :param A:       n x k binary matrix, if A is a 1 dim array it is assumed to be a column vector
    :param B:       k x m binary matrix, if B is a 1 dim array it is assumed to be a row vector

    :return:        n x m binary matrix, the Boolean matrix product of A and B
    """

    if len(A.shape) == 1 and len(B.shape) == 1:
        X = np.outer(A, B)

    elif len(A.shape) == 1 and len(B.shape) == 2:
        X = np.dot(A.reshape(len(A), 1), B)

    elif len(A.shape) == 2 and len(B.shape) == 1:
        X = np.dot(A, B.reshape(1, len(B)))

    else:
        X = np.dot(A, B)

    X[X > 0] = 1

    return X

def random_binary_matrix(n, m, rank, p_sparsity=0.50, p_noise=0.00, fixed_seed=0):
    """
    Generate a random binary matrix with specified Boolean rank, sparsity and noise

    :param n:           int, number of rows
    :param m:           int, number of columns
    :param rank:        int, max Boolean rank before noise is introduced
    :param p_sparsity:  float in [0,1], default=0.5, sparsity of X, as the probability of an entry x_ij is zero
    :param p_noise:     float in [0,1], default=0.0, noise in X, as the probability of an entry x_ij is flipped
    :param fixed_seed:  int, default=0, seed passed onto random number generator

    :return X:          n x m binary matrix X, that is at most p_noise * n * m squared Frobenius distance
                        away from an n x m binary matrix of Boolean rank=rank
    """

    # random number generation seed
    np.random.seed(fixed_seed)

    # sparsity
    q = 1 - np.sqrt(1 - p_sparsity**(1 / rank))  # probability of 0's in a Bernoulli trial
    p = 1 - q                                    # probability of 1's in a Bernoulli trial

    # initial X with Boolean rank at most = rank
    A = np.random.binomial(1, p, [n, rank])      # generating n x rank A with probability p for a 1
    B = np.random.binomial(1, p, [rank, m])
    X = boolean_matrix_product(A, B)

    # introducing noise
    for noise in range(int(p_noise * n * m)):    # at most p_noise * n * m entries perturbed in X
        i = np.random.choice(n)
        j = np.random.choice(m)                  # if same i, j comes up, noise is less

        X[i,j] = np.abs(X[i,j] - 1)

    return X

def preprocess(X):
    """
    Eliminate duplicate and zero rows/columns of a binary matrix but keep a record of them

    :param X:               n x m binary matrix with possibly missing entries set to np.nan

    Returns
    -------
    X_out :                 preprocessed matrix
    row_weights :           the number of times each unique row of X_out is repated in X
    col_weights :           the number of times each unique column of X_out is repated in X
    idx_rows_reconstruct :  X_out[idx_rows_reconstruct, :] puts row duplicates back
    idx_cols_reconstruct :  X_out[:, idx_cols_reconstruct] puts column duplicates back
    idx_zero_row :          index of zero row if any
    idx_zero_col :          index of zero col if any
    idx_rows_unique :       X[idx_rows_unique, :] eliminates duplicate rows of X
    idx_cols_unique :       X[:, idx_cols_unique] eliminates duplicate cols of X
    """

    X_in = np.copy(X)

    # if the matrix has missing entries, make them numerical so they can be subject to preprocessing
    if np.isnan(X_in).sum() > 0:
        is_missing = True
        X_in[np.isnan(X_in)] = -100
    else:
        is_missing = False

    #(1) operations on rows
    # delete row duplicates but record their indices and counts
    X_unique_rows, idx_rows_unique, idx_rows_reconstruct, row_weights = np.unique(X_in,
                                                                       return_index=True,
                                                                       return_inverse=True,
                                                                       return_counts=True,
                                                                       axis=0)
    is_zero_row = np.all(X_unique_rows == 0, axis=1)  # find the row of all zeros if it exists
    if is_zero_row.sum() > 0:
        X_unique_nonzero_rows = X_unique_rows[~is_zero_row]  # delete that row of zeros
        row_weights = row_weights[~is_zero_row]  # delete the count of zero rows
        idx_zero_row = np.where(is_zero_row)[0][0]  # get index of zero row
    else:
        X_unique_nonzero_rows = X_unique_rows
        idx_zero_row = None

    #(2) operations on columns
    # delete column duplicates but record their indices and counts
    X_unique_cols, idx_cols_unique, idx_cols_reconstruct, col_weights = np.unique(X_unique_nonzero_rows,
                                                                       return_index=True,
                                                                       return_inverse=True,
                                                                       return_counts=True,
                                                                       axis=1)

    is_zero_col = np.all(X_unique_cols == 0, axis=0)  # find the col of all zeros if it exists
    if is_zero_col.sum() > 0:
        X_unique_nonzero_cols = np.transpose(np.transpose(X_unique_cols)[~is_zero_col])
        # delete that col of zeros
        col_weights = col_weights[~is_zero_col]  # delete the count of zero columns
        idx_zero_col = np.where(is_zero_col)[0][0]  # store the index of zero column
    else:
        X_unique_nonzero_cols = X_unique_cols
        idx_zero_col = None
    X_out = X_unique_nonzero_cols

    # if the matrix had missing entries, turn these back to nan
    if is_missing:
        X_out[X_out == -100] = np.nan

    return X_out, row_weights, col_weights, idx_rows_reconstruct,\
           idx_cols_reconstruct, idx_zero_row, idx_zero_col, idx_rows_unique, idx_cols_unique

def un_preprocess(X, idx_rows_reconstruct, idx_zero_row, idx_cols_reconstruct, idx_zero_col):
    """
    Place back duplicate and zero rows and columns, the inverse fucntion of preprocess(X)

    :param X:                           preprocessed binary matrix
    :param idx_rows_reconstruct:        X_out[idx_rows_reconstruct, :] puts row duplicates back
    :param idx_zero_row:                index of zero row if any
    :param idx_cols_reconstruct:        X_out[:, idx_cols_reconstruct] puts column duplicates back
    :param idx_zero_col:                index of zero column if any

    :return X_out:                      binary matrix containing duplicate and zero rows and columns
    """

    #(1) operations on columns
    n, m = X.shape
    #(1.1) if there was a zero column removed add it back
    if idx_zero_col is None:
        X_zero_col = np.copy(X)
    else:
        X_zero_col = np.concatenate((X[:, 0:idx_zero_col],
                                     np.zeros([n, 1], dtype=int),
                                     X[:, idx_zero_col:m]), axis=1)
    # create duplicates of columns
    X_orig_cols = X_zero_col[:, idx_cols_reconstruct]

    #(2) operations on rows
    [n_tmp, m_tmp] = X_orig_cols.shape
    # if there was a zero row removed add it back
    if idx_zero_row is None:
        X_zero_row = X_orig_cols
    else:
        X_zero_row = np.concatenate((X_orig_cols[0:idx_zero_row, :],
                                     np.zeros([1, m_tmp], dtype=int),
                                     X_orig_cols[idx_zero_row:n_tmp, :]), axis=0)
    # create duplicates of rows
    X_out = X_zero_row[idx_rows_reconstruct, :]

    return X_out

def post_process_factorisation(A_in, B_in, idx_rows_reconstruct, idx_zero_row, idx_cols_reconstruct, idx_zero_col):
    """
    Place back duplicate and zero rows in A_in and duplicate and zero columns in B_in

    :param A_in:                        binary matrix to be extended with duplicate and zero rows
    :param B_in:                        binary matrix to be extended with duplicate and zero columns
    :param idx_rows_reconstruct:        A_in[idx_rows_reconstruct, :] puts row duplicates back
    :param idx_zero_row:                index of zero row if any
    :param idx_cols_reconstruct:        B_in[:, idx_cols_reconstruct] puts column duplicates back
    :param idx_zero_col:                index of zero column if any

    :return A:                          binary matrix with duplicate and zero rows
    :return B:                          binary matrix with duplicate and zero columns
    """

    #(1) operations on columns of B
    [n_tmp, m_tmp] = B_in.shape
    # if there was a zero column removed add it back
    if idx_zero_col is None:
        B_zero_col = np.copy(B_in)
    else:
        B_zero_col = np.concatenate((B_in[:, 0:idx_zero_col],
                                     np.zeros([n_tmp, 1], dtype=int),
                                     B_in[:, idx_zero_col:m_tmp]),
                                    axis=1)
    # create duplicates of columns
    B = B_zero_col[:, idx_cols_reconstruct]


    #(2) operations on rows of A
    [n_tmp, m_tmp] = A_in.shape
    # if there was a zero row removed add it back
    if idx_zero_row is None:
        A_zero_row = np.copy(A_in)
    else:
        A_zero_row = np.concatenate((A_in[0:idx_zero_row, :],
                                     np.zeros([1, m_tmp], dtype=int),
                                     A_in[idx_zero_row:n_tmp, :]),
                                    axis=0)
    # create duplicates of rows
    A = A_zero_row[idx_rows_reconstruct]

    return A, B


# heuristics
def BBQP_greedy_heur(Q_in, perturbed=False, transpose=False, revised=False, r_seed=None):
    """
    Greedy algorithm for the Bipartite Binary Quadratic Program: max_a,b a^T Q_in b where a, b are constrained binary
    Computing a rank-1 binary matrix ab^T that picks up the max weight of Q_in
    1. Sorts the rows of Q_in in decreasing order according to their sum of positive weights
    2. Aims to set a_i=1 for rows that increase the cumulative weight
    3. Sets b based on a

    :param Q_in:            n x m real matrix
    :param perturbed:       bool, default=False, perturb the original ordering of rows of Q_in
    :param transpose:       bool, default=False, use the transpose of Q_in
    :param revised:         bool, default=False, break ties in the original ordering by comparing negative sums
    :param r_seed:          int, default=None, use a random ordering

    :return a:              n dimensional binary vector
    :return b:              m dimensional binary vector
    """

    if transpose:
        # work on the transpose of the matrix
        Q = np.transpose(Q_in)
    else:
        Q = Q_in

    n, m = Q.shape

    if r_seed is None:
        # sum of positive weights in each row
        Q_pos = np.copy(Q)
        Q_pos[Q < 0] = 0
        w_pos = Q_pos.sum(axis=1)

        if perturbed:
            # slightly perturbed positive weights
            w_pos = w_pos * np.random.uniform(0.9, 1.1, n)

        if revised:
            # sum of negative weights in each row
            Q_neg = np.copy(Q)
            Q_neg[Q > 0] = 0
            w_neg = Q_neg.sum(axis=1)
            # sort w_pos in decreasing order, and resolve ties with sorting of w_neg in decreasing order
            sorted_i = np.lexsort((w_neg, w_pos))[::-1]
        else:
            # simply sort sort w_pos in decreasing order -- original ordering
            sorted_i = np.argsort(w_pos)[::-1]
    else:
        # use random ordering of rows
        np.random.seed(r_seed)
        sorted_i = np.random.permutation(n)

    a = np.zeros(n, dtype=int)
    s = np.zeros(m)
    for i in sorted_i:
        f_0 = np.sum(s[s >= 0])
        s_plus_Q = s + Q[i, :]
        f_1 = np.sum(s_plus_Q[s_plus_Q >= 0])
        if f_0 < f_1:
            a[i] = 1
            s = s + Q[i, :]

    b = np.zeros(m, dtype=int)
    b[s > 0] = 1

    if transpose:
        return b, a
    else:
        return a, b

def BBQP_alternating_heur(Q_in, a_in, b_in, transpose=False):
    """
    Alternating iterative algorithm for the Bipartite Binary Quadratic Program with a starting point a_in, b_in
    max_a,b a^T Q_in b where a, b are constrained binary
    Computing a rank-1 binary matrix ab^T that picks up the max weight of Q_in
    Sets a based on b_in or b based on a_in and then alternates

    :param Q_in:                        n x m real matrix
    :param a_in:                        n dimensional binary vector
    :param b_in:                        m dimensional binary vector
    :param transpose:                   bool, default=False, work on the transpose of Q_in

    :return a:                          n dimensional binary vector
    :return b:                          m dimensional binary vector
    """
    if transpose:
        # work on the transpose of the matrix
        Q = np.transpose(Q_in)
        a = b_in
        b = a_in
    else:
        Q = Q_in
        a = a_in
        b = b_in

    while True:
        idx_a = np.dot(Q, b) > 0
        a_new = np.zeros(Q.shape[0], dtype=int)
        a_new[idx_a] = 1
        if np.array_equal(a, a_new):
            break
        else:
            a = a_new

        idx_b = np.dot(a, Q) > 0
        b_new = np.zeros(Q.shape[1], dtype=int)
        b_new[idx_b] = 1
        if np.array_equal(b, b_new):
            break
        else:
            b = b_new

    if transpose:
        return b, a
    else:
        return a, b

def BBQP_mixed_heurs(Q, num_rand = 30):
    """
    Computes several variations of the Greedy and Alternating algorithms for the Bipartite Binary Quadratic Program

    :param Q_in:                        n x m real matrix
    :param num_rand:                    int, number of random ordering Greedy+Alternating algorithm to compute

    :return A:                          n x (8 + num_rand) binary matrix, each column for a different heur sol
    :return B:                          (8 + num_rand) x m binary matrix, each row for a different heur sol
    """

    (n,m) = Q.shape
    A = np.zeros((n, 8 + num_rand), dtype=int)
    B = np.zeros((8 + num_rand, m), dtype=int)

    # original
    a, b = BBQP_greedy_heur(Q)
    a, b = BBQP_alternating_heur(Q, a, b)
    A[:, 0] = a
    B[0, :] = b

    # original perturbed
    a, b = BBQP_greedy_heur(Q, perturbed=True)
    a, b = BBQP_alternating_heur(Q, a, b)
    A[:, 1] = a
    B[1, :] = b

    # original transpose
    a, b = BBQP_greedy_heur(Q, transpose=True)
    a, b = BBQP_alternating_heur(Q, a, b, transpose=True)
    A[:, 2] = a
    B[2, :] = b

    # original perturbed transpose
    a, b = BBQP_greedy_heur(Q, perturbed=True, transpose=True)
    a, b = BBQP_alternating_heur(Q, a, b, transpose=True)
    A[:, 3] = a
    B[3, :] = b

    # revised
    a, b = BBQP_greedy_heur(Q, revised=True)
    a, b = BBQP_alternating_heur(Q, a, b)
    A[:, 4] = a
    B[4, :] = b

    # revised perturbed
    a, b = BBQP_greedy_heur(Q, revised=True, perturbed=True)
    a, b = BBQP_alternating_heur(Q, a, b)
    A[:, 5] = a
    B[5, :] = b

    # revised transpose
    a, b = BBQP_greedy_heur(Q, revised=True, transpose=True)
    a, b = BBQP_alternating_heur(Q, a, b, transpose=True)
    A[:, 6] = a
    B[6, :] = b

    # revised perturbed transpose
    a, b = BBQP_greedy_heur(Q, revised=True, perturbed=True, transpose=True)
    a, b = BBQP_alternating_heur(Q, a, b, transpose=True)
    A[:, 7] = a
    B[7, :] = b

    # random
    for i in range(num_rand):
        a, b = BBQP_greedy_heur(Q, r_seed=i)
        a, b = BBQP_alternating_heur(Q, a, b)
        A[:, 8 + i] = a
        B[8 + i, :] = b

    return A, B

def BBQP_mixed_heur(Q, num_rand=30):
    """
    Computes several variations of the Greedy and Alternating algorithms for the Bipartite Binary Quadratic Program
    Returns the best one among them

    :param Q_in:                        n x m real matrix
    :param num_rand:                    int, number of random ordering Greedy+Alternating algorithm to compute

    :return a:                          n dimensional binary vector, with best objective value
    :return b:                          m dimensional binary vector, with best objective value
    """

    A, B = BBQP_mixed_heurs(Q, num_rand=num_rand)

    # compute objective of each heur sol
    obj = np.array([np.dot(A[:, l], np.dot(Q, B[l, :])) for l in range(A.shape[1])])

    l = np.argmax(obj)

    # print('BEST', Q_orig[Q_orig > 0].sum() - Q_orig[boolean_matrix_product(A[:, l], B[l, :]) == 1].sum())

    return A[:, l], B[l, :]

def BMF_k_greedy_heur(X, k, row_weights=None, col_weights=None,
                      mixed=True, revised=False, perturbed=False, transpose=False, r_seed=None):
    """
    Greedy algorithm for the rank-k Binary Matrix Factorisation problem
    Computes k rank-1 binary matrices sequentially which give the 1-best coverage of X with least 0-coverage
    Uses the greeady and alternating heuristics for Bipartite Binary Quadratic Programs as subroutine

    Parameters
    ----------
    X :                 n x m binary matrix with possibly missing entries set to np.nan
    k :                 int, rank of factorisation
    row_weights :       n dimensional array, weights to put on rows
    col_weights :       m dimensional array, weights to put on columns
    mixed:              bool, default=True,
                        use BBQP_mixed_heur(Q, num_rand=30) as subroutine
    perturbed:          bool, default=False,
                        use BBQP_greedy_heur(Q, perturbed=True) + alternating heuristic as subroutine
    transpose:          bool, default=False,
                        use BBQP_greedy_heur(Q, transpose=True) + alternating heuristic as subroutine
    revised:            bool, default=False,
                        use BBQP_greedy_heur(Q, revised=True) + alternating heuristic as subroutine
    r_seed:             int, default=None,
                        use BBQP_greedy_heur(Q, r_seed=r_seed) + alternating heuristic as subroutine

    Returns
    -------
    A:                  n x k  binary matrix
    B:                  k x m  binary matrix
    """
    (n, m) = X.shape

    if row_weights is None or col_weights is None:
        row_weights = np.ones(n, dtype=int)
        col_weights = np.ones(m, dtype=int)

    A = np.zeros([n, k], dtype=int)
    B = np.zeros([k, m], dtype=int)

    Q_sign = np.zeros((n, m), dtype=int)
    Q_sign[X == 1] = 1
    Q_sign[X == 0] = -1
    Weights = np.outer(row_weights, col_weights)
    Q_orig = Q_sign * Weights
    Q = np.copy(Q_orig)

    for i in range(k):
        if mixed:
            a, b = BBQP_mixed_heur(Q, 30)
        else:
            a, b = BBQP_greedy_heur(Q, perturbed, transpose, revised, r_seed)
            a, b = BBQP_alternating_heur(Q, a, b, transpose)
        A[:, i] = a
        B[i, :] = b
        idx_covered = boolean_matrix_product(A, B) == 1
        Q[idx_covered] = 0

        ## objective value in squared frobenius norm
        # print(Q_orig[Q_orig > 0].sum() - Q_orig[boolean_matrix_product(A, B) == 1].sum())

    return A, B

# column generation
def get_unique_rectangles(A, B):

    # eliminate identical rectangles
    A_unique, idx_A_rows_unique, idx_A_rows_reconstruct = np.unique(A,return_index=True,
                                                                    return_inverse=True,
                                                                    axis=1)

    B_unique, idx_B_cols_unique, idx_B_cols_reconstruct = np.unique(B,return_index=True,
                                                                    return_inverse=True,
                                                                    axis=0)

    # np.array_equal(Warm_A[:,idx_A_rows_unique[0]],WarmA_unique[:,0])

    [idx_A, idx_B] = np.unique(np.transpose([idx_A_rows_reconstruct, idx_B_cols_reconstruct]), axis=0).transpose()

    A_out = A_unique[:, idx_A]
    B_out = B_unique[idx_B, :]

    ## to double check
    # A_out = A_unique[:, idx_A]
    # B_out = B_unique[idx_B, :]
    # np.array_equal(boolean_matrix_product(A, B), boolean_matrix_product(A_out, B_out))

    return A_out, B_out

class ObjLimitCallback(cplex.callbacks.MIPInfoCallback):
    # create a new class that inherits from MIPInfoCallBack

    # this callback function aborts if a certain time limit has passed and we have a good enough solution
    # for a MAXIMISATION problem

    # override the method __call__ of the parent class as instructed in cplex documentation
    def __call__(self):
        if not self.aborted and self.has_incumbent():

            # get the objective of the incumbent solution
            obj = self.get_incumbent_objective_value()

            # get how much time has passed
            timeused = self.get_time() - self.starttime

            # if we spent enough time already and the objective is better than the goal
            if timeused > self.timelimit and obj > self.objgoal:
                print("Good enough solution at", timeused, "sec., obj =",
                      obj, ", quitting.")
                self.aborted = True
                self.abort()

class BMF_via_CG:
    """
    Class to compute rank-k binary matrix factorisation via column generation
    Works on exponential size integer program MIP and its LP relaxation MLP
    Computes k rank-1 binary matrices by generating rank-1 binary matrices when solving MLP via column generation,
        columns correspond to rank-1 binary matrices
    Then solves MIP over the rank-1 binary matrices generated by MLP to pick the best k of them

    Attributes
    ----------
    X :                     n x m binary matrix to be factorised, with possibly missing entries set to np.nan
    k :                     int, rank of factorisation
    A:                      n x k binary matrix, left hand side of k-BMF
    B:                      k x m binary matrix, right hand side of k-BMF

    row_weights :           n dimensional array of integers
                            all ones vector if X is not preprocessed
                            else, the number of times each unique row of preprocessed X is repated in original X
    col_weights :           m dimensional array of integers
                            all ones vector if X is not preprocessed
                            else, the number of times each unique column of preprocessed X is repated in original X
    idx_rows_reconstruct :  None if X not preprocessed,
                            else X[idx_rows_reconstruct, :] puts row duplicates back in preprocessed X
    idx_cols_reconstruct :  None, if X not preprocessed,
                            else X[:, idx_cols_reconstruct] puts column duplicates back in preprocessed X
    idx_zero_row :          index of zero row if any
    idx_zero_col :          index of zero col if any
    idx_rows_unique :       None, if X not preprocessed,
                            X[idx_rows_unique, :] eliminates duplicate rows of original X
    idx_cols_unique :       None, if X not preprocessed,
                            [:, idx_cols_unique] eliminates duplicate cols of original X

    MLP_A :                 n x p binary matrix, left hand side of rank-1 binary matrices generated during CG
    MLP_B :                 p x m binary matrix, right hand side of rank-1 binary matrices generated during CG
    MLP_model :             cplex model of MLP
    best_dual_bound :       best dual (lower) bound on MLP, consequently on MIP
    MLP_rel_gap :           current relative gap of MLP,
                            defined as (MLP_primal_obj - self.best_dual_bound)/ MLP_primal_obj
    MLP_abs_gap :           current absolute gap of MLP,
                            defined as MLP_primal_obj - self.best_dual_bound
    MLP_time :              cumulative total time spent on CG
    MLP_n_iter :            cumulative number of iterations in CG
    MLP_log :               pandas dataframe logging CG process

    MLP_pricer_IP :         cplex model of CG pricing IP

    """

    def __init__(self, X, k, A_init=None, B_init=None):
        """
        BMF_via_CG object constructor

        Parameters
        ----------
        X :         n x m binary matrix
        k :         int, rank of factorisation, k< min (n,m) otherwise problem is trivial
        A_init :    n x q binary matrix, default=None, specifies rows of warm start rank-1 binary matrices
        B_init :    q x m binary matrix, default=None, specifies columns of warm start rank-1 binary matrices

        Attributes
        ----------
        X :                     n x m binary matrix to be factorised, with possibly missing entries set to np.nan
        k :                     int, rank of factorisation
        A:                      n x k binary matrix, left hand side of k-BMF
        B:                      k x m binary matrix, right hand side of k-BMF

        row_weights :           n dimensional array of integers
                                all ones vector if X is not preprocessed
                                else, the number of times each unique row of preprocessed X is repated in original X
        col_weights :           m dimensional array of integers
                                all ones vector if X is not preprocessed
                                else, the number of times each unique column of preprocessed X is repated in original X
        idx_rows_reconstruct :  None if X not preprocessed,
                                else X[idx_rows_reconstruct, :] puts row duplicates back in preprocessed X
        idx_cols_reconstruct :  None, if X not preprocessed,
                                else X[:, idx_cols_reconstruct] puts column duplicates back in preprocessed X
        idx_zero_row :          index of zero row if any
        idx_zero_col :          index of zero col if any
        idx_rows_unique :       None, if X not preprocessed,
                                X[idx_rows_unique, :] eliminates duplicate rows of original X
        idx_cols_unique :       None, if X not preprocessed,
                                [:, idx_cols_unique] eliminates duplicate cols of original X

        MLP_A :                 n x p binary matrix, left hand side of rank-1 binary matrices generated during CG
        MLP_B :                 p x m binary matrix, right hand side of rank-1 binary matrices generated during CG
        MLP_model :             cplex model of MLP
        best_dual_bound :       best dual (lower) bound on MLP, consequently on MIP
        MLP_rel_gap :           current relative gap of MLP,
                                defined as (MLP_primal_obj - self.best_dual_bound)/ MLP_primal_obj
        MLP_abs_gap :           current absolute gap of MLP,
                                defined as MLP_primal_obj - self.best_dual_bound
        MLP_time :              cumulative total time spent on CG
        MLP_n_iter :            cumulative number of iterations in CG
        MLP_log :               pandas dataframe logging CG process

        MLP_pricer_IP :         cplex model of CG pricing IP

        """

        self.X = X
        self.k = k
        if k >= np.min(X.shape):
            raise ValueError(
                'k=%s >= min (%s, %s) dimension of X, so the rank-k binary matrix factorisation problem is trivial'
                % (k, X.shape[0], X.shape[1]))

        self.row_weights = np.ones(self.X.shape[0], dtype=int)
        self.col_weights = np.ones(self.X.shape[1], dtype=int)
        self.idx_rows_reconstruct, self.idx_cols_reconstruct, \
        self.idx_zero_row, self.idx_zero_col, \
        self.idx_rows_unique, self.idx_cols_unique = [None] * 6

        self.A = np.empty((X.shape[0], 0), dtype=int)
        self.B = np.empty((0, X.shape[1]), dtype=int)

        self.best_dual_bound = 0

        if A_init is not None and B_init is not None:

            if X.shape[0] != A_init.shape[0]:
                raise ValueError(
                    'Shapes of input X and A_init not aligned: %s (dim 0 of X) != %s (dim 0 of A_init)' % (
                    X.shape[0], A_init.shape[0]))
            elif X.shape[1] != B_init.shape[1]:
                raise ValueError(
                    'Shapes of input X and B_init not aligned: %s (dim 1 of X) != %s (dim 1 of B_init)' % (
                    X.shape[1], B_init.shape[1]))
            elif A_init.shape[1] != B_init.shape[0]:
                raise ValueError(
                    'Shapes of input A_init and B_init not aligned: %s (dim 1 of A_init) != %s (dim 0 of B_init)' %(
                        A_init.shape[1], B_init.shape[0]))
            else:
                self.MLP_A = A_init
                self.MLP_B = B_init
        else:
            self.MLP_A = np.empty((X.shape[0], 0), dtype=int)
            self.MLP_B = np.empty((0, X.shape[1]), dtype=int)

        self.MLP_model = None
        self.MLP_rel_gap = 100
        self.MLP_abs_gap = 1.0e10
        self.MLP_time = 0
        self.MLP_n_iter = 0
        self.MLP_log = None
        self.MLP_pricer_IP = None

    def preprocess_input_matrices(self):
        """
        Eliminate duplicate and zero rows/columns of X so a smaller equivalent problem can be solved
        """

        if self.idx_rows_reconstruct is not None:

            raise ValueError(
                'Problem has already been preprocessed')
        else:

            # preprocess X
            self.X, self.row_weights, self.col_weights, self.idx_rows_reconstruct, self.idx_cols_reconstruct,\
            self.idx_zero_row, self.idx_zero_col, self.idx_rows_unique, self.idx_cols_unique = preprocess(self.X)

            # preprocess MLP_A and MLP_B that hold MLP columns
            self.MLP_A = self.MLP_A[self.idx_rows_unique, :]
            self.MLP_B = self.MLP_B[:, self.idx_cols_unique]

            if self.idx_zero_row is not None:
                self.MLP_A = np.delete(self.MLP_A, self.idx_zero_row, axis=0)
            if self.idx_zero_col is not None:
                self.MLP_B = np.delete(self.MLP_B, self.idx_zero_col, axis=1)

            # preprocess A and B that is incumbent integer solution
            self.A = self.A[self.idx_rows_unique, :]
            self.B = self.B[:, self.idx_cols_unique]

            if self.idx_zero_row is not None:
                self.A = np.delete(self.A, self.idx_zero_row, axis=0)
            if self.idx_zero_col is not None:
                self.B = np.delete(self.B, self.idx_zero_col, axis=1)

    def post_process_output_matrices(self):
        """
        Place back duplicate and zero rows/columns so it corresponds to original input matrix
        """

        if self.idx_rows_reconstruct is None or \
                self.idx_cols_reconstruct is None or  \
                self.idx_rows_unique is None or \
                self.idx_cols_unique is None:
            raise ValueError(
                'Problem has not been preprocessed so it cannot be post-processed,\n use .preprocess_input_matrices() to preprocess')

        self.MLP_A, self.MLP_B = post_process_factorisation(self.MLP_A, self.MLP_B, self.idx_rows_reconstruct, self.idx_zero_row, self.idx_cols_reconstruct, self.idx_zero_col)

        self.A, self.B = post_process_factorisation(self.A, self.B, self.idx_rows_reconstruct, self.idx_zero_row, self.idx_cols_reconstruct, self.idx_zero_col)

        self.X = un_preprocess(self.X, self.idx_rows_reconstruct, self.idx_zero_row, self.idx_cols_reconstruct, self.idx_zero_col)

        # if matrices are set back to the original dimension, update the preprocessing paramaters to default too
        self.row_weights = np.ones(self.X.shape[0], dtype=int)
        self.col_weights = np.ones(self.X.shape[1], dtype=int)

        self.idx_rows_reconstruct, self.idx_cols_reconstruct, \
        self.idx_zero_row, self.idx_zero_col, \
        self.idx_rows_unique, self.idx_cols_unique = [None] * 6

    #specific to rho
    def MLP_set_cplex_model(self, rho):
        """
        Set up the master linear program MLP in cplex, variables corresponding to rank-1 binary matrices are added later

        :param rho:       float in (0, 1], the weight of covering a 0 each time it is covered
        """

        idx1 = np.transpose(np.where(self.X == 1))  # indices of 1-valued entries of X

        # initialise cplex object
        master = cplex.Cplex()

        # add a variable for each x_ij=1,  xi(i,j)=1 if x_ij=1 is not covered, xi(i,j)=0 if x_ij=1 is covered
        # and set its objective coefficient with the weight of x_ij
        xi_names = ["xi(%s,%s)" % (i, j) for (i, j) in idx1]
        master.variables.add(obj=[float(self.row_weights[i] * self.col_weights[j]) for (i, j) in idx1],
                             lb=[0] * len(idx1),
                             names=xi_names)

        # add a constraint for each x_ij=1.
        # later on for each rectangle variable created the corresponding column will be filled in
        master.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["xi(%s,%s)" % (i, j)],
                                       val=[1])
                      for (i, j) in idx1
                      ],
            senses=["G"] * len(idx1),
            rhs=[1] * len(idx1),
            names=["con(%s,%s)" % (i, j) for (i, j) in idx1])

        # add the constraint that we cannot use more thank k rectangles, columns will be filled in later
        master.linear_constraints.add(
            lin_expr=[cplex.SparsePair()],
            rhs=[-self.k],
            senses=["G"],
            names=["-sum rectangles >= -k"]
        )

        self.MLP_model = master
        self.MLP_model.rect_names = []
        self.MLP_model.rho = rho

    #specific to rho
    def MLP_pricer_IP_set_cplex_model(self):
        """
        Set up pricing problem of MLP.
        The constraints and variables in this problem always stay the same but the objective function changes
        during the column generation loop.
        """

        idx1 = np.transpose(np.where(self.X == 1))  # indices of 1-valued entries of X
        idx0 = np.transpose(np.where(self.X == 0))  # indices of 0-valued entries of X
        (n, m) = self.X.shape

        pricer_IP = cplex.Cplex()
        pricer_IP.set_problem_name('pricer_IP')

        a_names = ["a(%s)" % i for i in range(n)]
        pricer_IP.variables.add(types=[pricer_IP.variables.type.binary] * n,
                                  names=a_names)

        b_names = ["b(%s)" % j for j in range(m)]
        pricer_IP.variables.add(types=[pricer_IP.variables.type.binary] * m,
                                  names=b_names)

        y_names = ["y(%s,%s)" % (i, j) for i in range(n) for j in range(m)]
        pricer_IP.variables.add(lb=[0] * n * m,
                                  ub=[1] * n * m,
                                  names=y_names)

        pricer_IP.objective.set_linear(
            ["y(%s,%s)" % (i, j), self.MLP_model.rho * float(-1 * self.row_weights[i] * self.col_weights[j])]
            for [i, j] in idx0)

        pricer_IP.objective.set_sense(pricer_IP.objective.sense.maximize)

        # a_i +b_j - 1 <= y_ij for x_ij=0
        pricer_IP.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=["a(%s)" % i, "b(%s)" % j, "y(%s,%s)" % (i, j)],
                                 val=[1, 1, -1])
                for (i, j) in idx0
            ],
            rhs=[1] * len(idx0),
            senses=["L"] * len(idx0),
            names=["a(%s) + b(%s) - y(%s,%s) <= 1" % (i, j, i, j) for (i, j) in idx0]
        )

        # y_ij <= a_i
        pricer_IP.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=["y(%s,%s)" % (i, j), "a(%s)" % i],
                                 val=[1, -1])
                for [i, j] in idx1
            ],
            rhs=[0] * len(idx1),
            senses=["L"] * len(idx1),
            names=["y(%s,%s) <= a(%s)" % (i, j, i) for [i, j] in idx1]
        )

        # y_ij <= b_j
        pricer_IP.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=["y(%s,%s)" % (i, j), "b(%s)" % j],
                                 val=[1, -1])
                for [i, j] in idx1
            ],
            rhs=[0] * len(idx1),
            senses=["L"] * len(idx1),
            names=["y(%s,%s) <= b(%s)" % (i, j, j) for [i, j] in idx1]
        )

        self.MLP_pricer_IP = pricer_IP
        self.MLP_pricer_IP.a_names = a_names
        self.MLP_pricer_IP.b_names = b_names
        self.MLP_pricer_IP.y_names = y_names

    #specific to rho
    def MLP_add_columns(self, A, B):
        """
        Add the rank-1 binary matrices A[:, l].B[l, :] to MLP

        :param A:       n x p binary matrix
        :param B:       p x m binary matrix
        """

        rect_num = A.shape[1]

        if rect_num > 0:

            rectangles = [np.outer(A[:, b], B[b, :]) for b in range(rect_num)]

            rectangles_scaled = [np.dot(np.dot(np.diag(self.row_weights), rectangle), np.diag(self.col_weights)) for rectangle in rectangles]

            rect_objs = [self.MLP_model.rho * float(rectangle_scaled[self.X == 0].sum()) for rectangle_scaled in rectangles_scaled]

            rect_names = ["r(%s)" % b for b in range(len(self.MLP_model.rect_names), len(self.MLP_model.rect_names) + rect_num)]

            rect_cols = [[self.MLP_model.linear_constraints.get_names(),rectangle[self.X == 1].tolist() + [-1]] for rectangle in rectangles]

            self.MLP_model.variables.add(obj = rect_objs,
                                         names = rect_names,
                                         columns = rect_cols,
                                         lb = [0] * rect_num)

            self.MLP_model.rect_names = self.MLP_model.rect_names + rect_names

    #specific to rho
    def MLP_initialise_duals(self):

        n, m = self.X.shape
        idx0 = np.transpose(np.where(self.X == 0))  # indices of 0-valued entries of X

        self.MLP_model.H = np.zeros((n, m))
        self.MLP_model.H[self.X == 0] = [-1 * self.MLP_model.rho * self.row_weights[i] * self.col_weights[j] for (i, j) in idx0]

        self.MLP_model.mu = None

    #specific to rho
    def MLP_update_duals(self):

        idx1 = np.transpose(np.where(self.X == 1))  # indices of 1-valued entries of X

        self.MLP_model.H[self.X == 1] = [self.MLP_model.solution.get_dual_values("con(%s,%s)" % (i, j)) for [i, j] in idx1]
        self.MLP_model.mu = self.MLP_model.solution.get_dual_values("-sum rectangles >= -k")

    #specific to rho
    def MLP_update_best_dual_bound(self, round_dual_bound):
        """
        Compute the current dual bound on MLP and update the best dual bound if it improved

        :param round_dual_bound:       bool, default=True, take ceiling of dual bound or not
        """

        # the dual bound with the current bound on PP
        current_dual_bound = self.MLP_model.solution.get_objective_value() - self.k * self.MLP_pricer_IP.solution.MIP.get_best_objective()
        self.best_dual_bound = max([self.best_dual_bound, current_dual_bound])

        #if we are told to round the dual bound
        if round_dual_bound:
            self.best_dual_bound = ceil(np.around(self.best_dual_bound, 2))

    #generic
    def MLP_update_gaps(self):

        primal = self.MLP_model.solution.get_objective_value()
        dual = self.best_dual_bound
        self.MLP_rel_gap = max( [0, 100 * ( primal - dual) / (1.e-10 + primal)])

        self.MLP_abs_gap = max([0, primal - dual])

    #generic
    def MLP_heuristic_pricing(self, num_rand=22):
        """
        Compute 30 heuristic solutions to the pricing probelem, return unique ones sorted

        :param num_rand:       int, default=22, number of random ordering solutions to compute
        """

        A, B = BBQP_mixed_heurs(self.MLP_model.H, num_rand=num_rand)

        # different variations of greedy may give the same output, eliminate identical rectangles
        A, B = get_unique_rectangles(A, B)

        # compute objective of each greedy sol
        obj = np.array([np.dot(A[:, l], np.dot(self.MLP_model.H, B[l, :])) for l in range(A.shape[1])])

        # decreasing order
        obj_sort_idx = np.argsort(obj)[::-1]
        A_heur = A[:, obj_sort_idx]
        B_heur = B[obj_sort_idx, :]
        obj_heur = obj[obj_sort_idx]

        return A_heur, B_heur, obj_heur

    def MLP_exact_pricing(self, a, b, n_iter, start_time, max_time, min_price_time, objgoal):
        """
        Solve the pricing problem IP via CPLEX

        Parameters
        ----------
        a:                  n dimensional vector, left hand side of warm start best heuristic rank-1 bianry matrix
        b:                  m dimensional vector, right hand side of warm start best heuristic rank-1 bianry matrix
        n_iter:             int, current interation in column generation loop
        start_time:         when column generation started
        max_time:           max time of column generation
        min_price_time:     min seconds before we can stop pricing IP solving even if we have a negative reduced cost sol
        objgoal:            if this goal is reached then we can stop solving the pricing IP

        """

        n, m = self.X.shape
        idx1 = np.transpose(np.where(self.X == 1))  # indices of 1-valued entries of X

        # add heuristic solution as starting point to CPLEX
        self.MLP_pricer_IP.MIP_starts.add(cplex.SparsePair(self.MLP_pricer_IP.a_names +
                                                       self.MLP_pricer_IP.b_names +
                                                       self.MLP_pricer_IP.y_names,
                                                       a.flatten().tolist() +
                                                       b.flatten().tolist() +
                                                       np.outer(a, b).flatten().tolist()
                                                       ),
                                      self.MLP_pricer_IP.MIP_starts.effort_level.auto,
                                      'heur_sol_at_iteration_%s' % n_iter)

        # set the objective coefficients of the pricing problem from dual values of master
        self.MLP_pricer_IP.objective.set_linear(["y(%s,%s)" % (i, j), self.MLP_model.H[i, j]] for [i, j] in idx1)

        # set the objective offset of the pricing problem from dual values of master
        self.MLP_pricer_IP.objective.set_offset(-1 * self.MLP_model.mu)

        # timelimit on pricing problem is what's left of max_time
        self.MLP_pricer_IP.parameters.timelimit.set(max(5, max_time - (self.MLP_model.get_time() - start_time)))

        # add the callback which will stop the exact pricing if we have a good enough solution
        objlimit_cb = self.MLP_pricer_IP.register_callback(ObjLimitCallback)
        objlimit_cb.timelimit = min_price_time
        objlimit_cb.aborted = False
        objlimit_cb.starttime = self.MLP_pricer_IP.get_time()
        objlimit_cb.objgoal = objgoal

        pp_start_time = self.MLP_pricer_IP.get_time()
        self.MLP_pricer_IP.solve()
        self.MLP_pricer_IP.time = self.MLP_pricer_IP.get_time() - pp_start_time

        a_cplex_float = np.array(self.MLP_pricer_IP.solution.get_values(self.MLP_pricer_IP.a_names)).reshape(n, 1)
        b_cplex_float = np.array(self.MLP_pricer_IP.solution.get_values(self.MLP_pricer_IP.b_names)).reshape(1, m)
        A_cplex = np.zeros([n, 1], dtype=int)
        A_cplex[a_cplex_float > 1. - EPS] = 1
        B_cplex = np.zeros([1, m], dtype=int)
        B_cplex[b_cplex_float > 1. - EPS] = 1

        return A_cplex, B_cplex

    #generic
    def MLP_update_columns(self, A, B):

        self.MLP_A = np.concatenate((self.MLP_A, A), axis=1)
        self.MLP_B = np.concatenate((self.MLP_B, B), axis=0)

    #generic
    def MLP_set_cplex_params(self):

        # turn the display off for now
        self.MLP_model.set_log_stream(None)
        self.MLP_model.set_results_stream(None)

        # set to use interior point method rather then simplex
        # 'method for linear optimization  :\n  0 = automatic\n  1 = primal simplex\n  2 = dual simplex\n
        # 3 = network simplex\n  4 = barrier\n  5 = sifting\n  6 = concurrent optimizers'
        self.MLP_model.parameters.lpmethod.set(4)  # 4

        # also tell cplex that once barrier finishes it should not convert the solution to a vertex solution
        # 'solution information CPLEX will attempt to compute  :\n  0 = auto\n
        # 1 = basic solution\n  2 = primal dual vector pair'
        self.MLP_model.parameters.solutiontype.set(2)  # 2

    def MLP_print_report(self, n_iter, start_time):

        print("\nAt iteration %.0f, at %.0f sec" % (n_iter, self.MLP_model.get_time() - start_time))
        print("The RMLP objective is:\t\t\t\t\t\t\t%.5f" % self.MLP_model.solution.get_objective_value())
        print("The bound on MLP is:\t\t\t\t\t\t\t%.5f" % self.best_dual_bound)
        print('The optimality gap for MLP is:\t\t\t\t\t%.2f' % (self.MLP_rel_gap))
        print("The number of columns in RMLP is:\t\t\t\t%.0f \n" % self.MLP_A.shape[1])

    def MLP_log_report(self):

        self.MLP_log.loc[self.MLP_n_iter, 'MLP_time'] = self.MLP_time
        self.MLP_log.loc[self.MLP_n_iter, 'MLP_primal_obj'] = self.MLP_model.solution.get_objective_value()
        self.MLP_log.loc[self.MLP_n_iter, 'MLP_dual_obj'] = self.best_dual_bound
        self.MLP_log.loc[self.MLP_n_iter, 'MLP_rel_gap'] = self.MLP_rel_gap
        self.MLP_log.loc[self.MLP_n_iter, 'MLP_n_cols'] = self.MLP_A.shape[1]

    def MLP_IP_pricer_log_report(self):

            self.MLP_log.loc[self.MLP_n_iter, 'pricer_IP_time'] = self.MLP_pricer_IP.time
            self.MLP_log.loc[self.MLP_n_iter, 'pricer_IP_obj'] = -1 * self.MLP_pricer_IP.solution.get_objective_value()

    def MLP_solve(self, rho=1, max_time=1000, min_price_time=25, display=True, log=False,
                  round_dual_bound=True, is_exact_pricing=False, is_dual_focus_pricing=False,
                  relative_gap=TOL, abs_gap = TOL, n_col_add_per_iter=2):

        """
        Column Generation (CG) method to solve the LP relaxation of MIP
        Generates rank-1 binary matrices, to access the matrices generated get self.MLP_A, self.MLP_B

        Parameters
        ----------
        rho :                       float in (0, 1], the weight of covering a 0 each time it is covered
                                    if =1, solves MLP(1)
                                    if =1/k, solves MLP(1/k)=MLP_F LP relaxation under squared Frobenius objective
        max_time :                  maximum seconds to spend on CG, default=1000
        min_price_time :            minimum seconds to solve pricing problem IP
        display :                   bool, default=True, to display progress or no
        round_dual_bound:           bool, default=True, take ceiling of dual bound or not
        is_exact_pricing:           bool, default=False, at every iteration solve the pricing IP via Cplex
                                    -- very time consuming but helps obtain stronger dual bound
        is_dual_focus_pricing:      bool, default=False, actively try to improve the dual bound
        relative_gap:               float in [0, 1] default=TOL,
                                    stop CG if (MLP_primal_obj - best_MLP_dual_obj) / MLP_primal_obj < relative_gap
        abs_gap:                    bool, default=TOL,
                                    stop CG if (MLP_primal_obj - best_MLP_dual_obj) < abs_gap
        n_col_add_per_iter:         int, default=2,
                                    max number of rank-1 binary matrices to be added to MLP at each iteration

        """
        # if previously some columns were generated by a different objective, reset the dual bound to current obj function
        if self.MLP_model is not None:
            if self.MLP_model.rho != rho:
                self.best_dual_bound = 0
        # populate self.MLP_model
        self.MLP_set_cplex_model(rho)
        self.MLP_set_cplex_params()
        # add initial columns in A,B to master LP
        self.MLP_add_columns(self.MLP_A, self.MLP_B)
        # populate self.MLP_pricer_IP
        self.MLP_pricer_IP_set_cplex_model()
        if display is False:
            self.MLP_pricer_IP.set_log_stream(None)
            self.MLP_pricer_IP.parameters.mip.display.set(0)
        # initialise self.MLP_model.H
        self.MLP_initialise_duals()
        # initialise self.MLP_log
        if log and self.MLP_log is None:
            self.MLP_log = pd.DataFrame()

        prev_time = self.MLP_time
        start_time = self.MLP_model.get_time()
        n_iter = 0
        MLP_obj_record = 0

        if display:
            print('\n\nSolving MLP(%s) via column generation\n' % rho)

        # column generation procedure
        while True:
            n_iter += 1
            self.MLP_n_iter += 1

            # solve current RMLP
            self.MLP_model.solve()

            if display:
                self.MLP_print_report(n_iter, start_time)
            self.MLP_time = prev_time + self.MLP_model.get_time() - start_time
            if log:
                self.MLP_log_report()

            # stop column generation if max_time reached
            if self.MLP_model.get_time() - start_time > max_time:
                break

            # update record of master dual values
            self.MLP_update_duals()

            # solve 30 variations of greedy+alt, sorted by best first
            A_heur, B_heur, obj_heur = self.MLP_heuristic_pricing()

            if display:
                print('The heuristic pricing problem objective is:\t\t%.5f\n' % (-1 * (obj_heur[0] - self.MLP_model.mu)))

            # check for which heuristic rectangles we have (a^T H b) > mu, so negative reduced cost
            idx_pos = np.where(obj_heur > self.MLP_model.mu + TOL)[0]

            # what objective value of the pricing problem could decrease the duality gap?
            if is_exact_pricing or \
                    (self.MLP_model.solution.get_objective_value() - self.best_dual_bound)/ (self.MLP_model.solution.get_objective_value() + 1.e-10) < 0.01:
                pricing_objective_goal = 0.99 * (self.MLP_model.solution.get_objective_value() - self.best_dual_bound) / self.k
            else:
                pricing_objective_goal = TOL

            # start solving PP via cplex if
            # 1. doing exact pricing or
            # 2. no heursitic rank-1 matrix has negative reduced cost
            # 3. or trying to close duality gap and heuristic solution of PP suggests we could improve dual bound
            if is_exact_pricing or \
                    obj_heur[0] - self.MLP_model.mu < TOL or \
                    (is_dual_focus_pricing and obj_heur[0] - self.MLP_model.mu < pricing_objective_goal) or \
                    ((n_iter % 50 == 0 and n_iter > 200) and abs(MLP_obj_record - self.MLP_model.solution.get_objective_value()) < 1):

                used_cplex = True
                A_cplex, B_cplex = self.MLP_exact_pricing(A_heur[:, 0], B_heur[0, :], n_iter, start_time, max_time, min_price_time, pricing_objective_goal)

                if display:
                    print('\n%s' % self.MLP_pricer_IP.solution.status[self.MLP_pricer_IP.solution.get_status()])
                    print("The cplex pricing problem objective is:\t\t\t%.5f" % (
                                -1 * self.MLP_pricer_IP.solution.get_objective_value()))

                self.MLP_update_best_dual_bound(round_dual_bound)
                self.MLP_update_gaps()

                if log:
                    self.MLP_IP_pricer_log_report()

                # stop column generation if we have optimality or optimal dual bound found for master IP
                if (self.MLP_pricer_IP.solution.get_objective_value() < 0. + TOL) or \
                        (self.best_dual_bound >= self.MLP_model.solution.get_objective_value()) or \
                        (self.MLP_rel_gap <= relative_gap) or \
                        (self.MLP_abs_gap <= abs_gap):
                    self.MLP_time = prev_time + self.MLP_model.get_time() - start_time
                    if display:
                        self.MLP_print_report(n_iter, start_time)
                    if log:
                        self.MLP_log_report()
                    break
            else:
                used_cplex = False

            if used_cplex:
                A_new = A_cplex
                B_new = B_cplex
            else:
                A_new = A_heur[:, idx_pos]
                B_new = B_heur[idx_pos, :]

                # only keep as many as we can add to MLP
                A_new = A_new[:, :min([n_col_add_per_iter, A_new.shape[1]])]
                B_new = B_new[:min([n_col_add_per_iter, B_new.shape[0]]), :]

            if n_iter % 10 == 0:
                MLP_obj_record = self.MLP_model.solution.get_objective_value()

            self.MLP_add_columns(A_new, B_new)
            self.MLP_update_columns(A_new, B_new)


    def MIP_set_cplex_model(self, rho):
        """
        Set up the master integer program MIP(rho) in cplex

        :param rho:       float in (0, 1], the weight of covering a 0 each time it is covered
        """

        MLP_n_cols = self.MLP_A.shape[1]

        if MLP_n_cols == 0:
            raise ValueError(
                'There are no columns available to be added to MIP')

        # indices of 1-valued entries of X, these are the 'edges'
        idx1 = np.transpose(np.where(self.X == 1))

        # initialise cplex object
        self.MIP_model = cplex.Cplex()

        # add a variable for each edge e, xi(e)=1 if edge e is not covered, xi(e)=0 if edge e is covered
        xi_names = ["xi(%s,%s)" % (i, j) for (i, j) in idx1]
        self.MIP_model.variables.add(obj=[float(self.row_weights[i] * self.col_weights[j]) for (i, j) in idx1],
                             names=xi_names)

        # add a constraint for each edge
        self.MIP_model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["xi(%s,%s)" % (i, j)],
                                       val=[1])
                      for (i, j) in idx1
                      ],
            senses=["G"] * len(idx1),
            rhs=[1] * len(idx1),
            names=["con(%s,%s)" % (i, j) for (i, j) in idx1])

        # add the constraint that we cannot use more thank k rectangles, columns will be filled in later
        self.MIP_model.linear_constraints.add(
            lin_expr=[cplex.SparsePair()],
            rhs=[-self.k],
            senses=["G"],
            names=["-sum rectangles >= -k"]
        )

        rectangles = [np.outer(self.MLP_A[:, b], self.MLP_B[b, :]) for b in range(MLP_n_cols)]

        rectangles_scaled = [np.dot(np.dot(np.diag(self.row_weights), rectangle), np.diag(self.col_weights)) for rectangle in rectangles]

        rect_objs = [rho * float(rectangle_scaled[self.X == 0].sum()) for rectangle_scaled in rectangles_scaled]

        rect_names = ["r(%s)" % b for b in range(MLP_n_cols)]

        rect_cols = [[self.MIP_model.linear_constraints.get_names(),rectangle[self.X == 1].tolist() + [-1]] for rectangle in rectangles]

        self.MIP_model.variables.add(obj = rect_objs,
                                     names = rect_names,
                                     columns = rect_cols,
                                     lb = [0] * MLP_n_cols,
                                     ub = [1] * MLP_n_cols,
                                     types = [self.MIP_model.variables.type.binary] * MLP_n_cols)

        self.MIP_model.rect_names = rect_names

    def MIP_frob_set_cplex_model(self):
        """
        Set up the master integer program MIP_F in cplex which has the squared Frobenius norm objective function

        """

        MLP_n_cols = self.MLP_A.shape[1]

        if MLP_n_cols == 0:
            raise ValueError(
                'There are no columns available to be added to MIP')

        # indices of 1-valued entries of X, these are the 'edges'
        idx1 = np.transpose(np.where(self.X == 1))
        idx0 = np.transpose(np.where(self.X == 0))

        # initialise cplex object
        self.MIP_model = cplex.Cplex()

        # add a variable for each edge e, xi(e)=1 if edge e is not covered, xi(e)=0 if edge e is covered
        xi_names = ["xi(%s,%s)" % (i, j) for (i, j) in idx1]
        self.MIP_model.variables.add(obj=[float(self.row_weights[i] * self.col_weights[j]) for (i, j) in idx1],
                             names=xi_names)

        pi_names = ["pi(%s,%s)" % (i, j) for (i, j) in idx0]
        self.MIP_model.variables.add(obj=[float(self.row_weights[i] * self.col_weights[j]) for (i, j) in idx0],
                             names=pi_names,
                             types=[self.MIP_model.variables.type.binary] * len(idx0))

        # add a constraint for each edge.
        # later on for each biclique variable created the corresponding column will be filled in
        self.MIP_model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["xi(%s,%s)" % (i, j)],
                                       val=[1])
                      for (i, j) in idx1
                      ],
            senses=["G"] * len(idx1),
            rhs=[1] * len(idx1),
            names=["con(%s,%s)" % (i, j) for (i, j) in idx1])

        self.MIP_model.linear_constraints.add(
            lin_expr=[cplex.SparsePair(ind=["pi(%s,%s)" % (i, j)],
                                       val=[self.k])
                      for (i, j) in idx0
                      ],
            senses=["G"] * len(idx0),
            rhs=[0] * len(idx0),
            names=["con(%s,%s)" % (i, j) for (i, j) in idx0])

        # add the constraint that we cannot use more thank k rectangles
        self.MIP_model.linear_constraints.add(
            lin_expr=[cplex.SparsePair()],
            rhs=[-self.k],
            senses=["G"],
            names=["-sum rectangles >= -k"]
        )

        rectangles = [np.outer(self.MLP_A[:, b], self.MLP_B[b, :]) for b in range(MLP_n_cols)]

        rectangles_scaled = [np.dot(np.dot(np.diag(self.row_weights), rectangle), np.diag(self.col_weights)) for rectangle in
                             rectangles]

        rect_objs = [0 for _ in rectangles_scaled]

        rect_names = ["r(%s)" % b for b in range(MLP_n_cols)]

        rect_cols = [[self.MIP_model.linear_constraints.get_names(),
                     rectangle[self.X == 1].tolist() + (-1 * rectangle[self.X == 0]).tolist() + [-1]] for rectangle in
                    rectangles]

        self.MIP_model.variables.add(obj=rect_objs,
                                     names=rect_names,
                                     columns=rect_cols,
                                     lb=[0] * MLP_n_cols,
                                     ub=[1] * MLP_n_cols,
                                     types=[self.MIP_model.variables.type.binary] * MLP_n_cols)

        self.MIP_model.rect_names = rect_names

    def MIP_solve(self, objective=1., max_time=300, display=2):
        """
        Set up the master integer program MIP in cplex

        :param objective:       float in (0, 1] or 'frobenius'
                                if float, solves model MIP(rho) with weight rho on the 0s
                                if 'frobenius', solves model MIP_F squared Frobenius norm objective function
        :param max_time:        max seconds to spend on solving MIP
        :param setting:         cplex parameters.mip.display setting, see cplex doc

        """

        if objective == 'frobenius':
            self.MIP_frob_set_cplex_model()
        elif isinstance(objective, int) or isinstance(objective, float):
            self.MIP_set_cplex_model(objective)
        else:
            raise ValueError(
                'Objective parameter input: %s is not equal to "frobenius" or not of numeric type' % (objective))

        self.MIP_model.parameters.timelimit.set(max_time)
        self.MIP_model.parameters.mip.display.set(display)

        self.MIP_model.solve()

        print(self.MIP_model.solution.status[self.MIP_model.solution.get_status()])

        # extract the rectangles chosen to be in the rank-k factorisation
        MIP_sol = np.array(self.MIP_model.solution.get_values(self.MIP_model.rect_names))
        rects_used = np.arange(len(self.MIP_model.rect_names))[MIP_sol > (1. - EPS)]

        if len(rects_used) > self.k:
            raise ValueError('MIP selected more than k rectangles, set EPS smaller to avoid numerical errors')

        self.A = np.zeros([self.X.shape[0], self.k], dtype=int)
        self.B = np.zeros([self.k, self.X.shape[1]], dtype=int)
        for l in range(len(rects_used)):
            self.A[:, l] = self.MLP_A[:, rects_used[l]]
            self.B[l, :] = self.MLP_B[rects_used[l], :]

# compact IP
class BMF_via_compact_IP:
    """
    Class to compute rank-k binary matrix factorisation via compact integer program CIP
    Solves polynomial size model CIP via CPLEX to get k rank-1 binary matrices

    Attributes
    ----------
    X :                     n x m binary matrix to be factorised, with possibly missing entries set to np.nan
    k :                     int, rank of factorisation
    A:                      n x k binary matrix, left hand side of k-BMF
    B:                      k x m binary matrix, right hand side of k-BMF

    row_weights :           n dimensional array of integers
                            all ones vector if X is not preprocessed
                            else, the number of times each unique row of preprocessed X is repated in original X
    col_weights :           m dimensional array of integers
                            all ones vector if X is not preprocessed
                            else, the number of times each unique column of preprocessed X is repated in original X
    idx_rows_reconstruct :  None if X not preprocessed,
                            else X[idx_rows_reconstruct, :] puts row duplicates back in preprocessed X
    idx_cols_reconstruct :  None, if X not preprocessed,
                            else X[:, idx_cols_reconstruct] puts column duplicates back in preprocessed X
    idx_zero_row :          index of zero row if any
    idx_zero_col :          index of zero col if any
    idx_rows_unique :       None, if X not preprocessed,
                            X[idx_rows_unique, :] eliminates duplicate rows of original X
    idx_cols_unique :       None, if X not preprocessed,
                            [:, idx_cols_unique] eliminates duplicate cols of original X

    CIP_model :             cplex model of compact integer program CIP
    """

    def __init__(self, X, k, A_init=None, B_init=None):
        """
        BMF_via_compact_IP object constructor

        Parameters
        ----------
        X :         n x m binary matrix
        k :         int, rank of factorisation, k< min (n,m) otherwise problem is trivial
        A_init :    n x p binary matrix, default=None, specifies rows of warm start rank-1 binary matrices,
                    needs p <=k otherwise only first k rank-1 binary matrices used
        B_init :    p x m binary matrix, default=None, specifies columns of warm start rank-1 binary matrices
                    needs p <=k otherwise only first k rank-1 binary matrices used

        Attributes
        ----------
        X :                     n x m binary matrix to be factorised, with possibly missing entries set to np.nan
        k :                     int, rank of factorisation
        A:                      n x k binary matrix, left hand side of k-BMF
        B:                      k x m binary matrix, right hand side of k-BMF

        row_weights :           n dimensional array of integers
                                all ones vector if X is not preprocessed
                                else, the number of times each unique row of preprocessed X is repated in original X
        col_weights :           m dimensional array of integers
                                all ones vector if X is not preprocessed
                                else, the number of times each unique column of preprocessed X is repated in original X
        idx_rows_reconstruct :  None if X not preprocessed,
                                else X[idx_rows_reconstruct, :] puts row duplicates back in preprocessed X
        idx_cols_reconstruct :  None, if X not preprocessed,
                                else X[:, idx_cols_reconstruct] puts column duplicates back in preprocessed X
        idx_zero_row :          index of zero row if any
        idx_zero_col :          index of zero col if any
        idx_rows_unique :       None, if X not preprocessed,
                                X[idx_rows_unique, :] eliminates duplicate rows of original X
        idx_cols_unique :       None, if X not preprocessed,
                                [:, idx_cols_unique] eliminates duplicate cols of original X

        CIP_model :             cplex model of compact integer program CIP
        """

        self.X = X
        if k >= np.min(X.shape):
            raise ValueError(
                'k=%s >= min (%s, %s) dimension of X, so the rank-k binary matrix factorisation problem is trivial'
                % (k, X.shape[0], X.shape[1]))
        self.k = k

        self.row_weights = np.ones(self.X.shape[0], dtype=int)
        self.col_weights = np.ones(self.X.shape[1], dtype=int)
        self.idx_rows_reconstruct, self.idx_cols_reconstruct, \
        self.idx_zero_row, self.idx_zero_col, \
        self.idx_rows_unique, self.idx_cols_unique = [None] * 6

        if A_init is not None and B_init is not None:

            if X.shape[0] != A_init.shape[0]:
                raise ValueError(
                    'Shapes of input X and A_init not aligned: %s (dim 0 of X) != %s (dim 0 of A_init)' % (
                    X.shape[0], A_init.shape[0]))
            elif X.shape[1] != B_init.shape[1]:
                raise ValueError(
                    'Shapes of input X and B_init not aligned: %s (dim 1 of X) != %s (dim 1 of B_init)' % (
                    X.shape[1], B_init.shape[1]))
            elif A_init.shape[1] != B_init.shape[0]:
                raise ValueError(
                    'Shapes of input A_init and B_init not aligned: %s (dim 1 of A_init) != %s (dim 0 of B_init)' %(
                        A_init.shape[1], B_init.shape[0]))
            else:
                self.A = A_init
                self.B = B_init
        else:
            self.A = np.empty((X.shape[0], 0), dtype=int)
            self.B = np.empty((0, X.shape[1]), dtype=int)

        self.CIP_model = None

    def preprocess_input_matrices(self):
        """
        Eliminate duplicate and zero rows/columns of X so a smaller equivalent problem can be solved
        """

        if self.idx_rows_reconstruct is not None:

            raise ValueError(
                'Problem has already been preprocessed')
        else:
            # preprocess X
            self.X, self.row_weights, self.col_weights, self.idx_rows_reconstruct, self.idx_cols_reconstruct,\
            self.idx_zero_row, self.idx_zero_col, self.idx_rows_unique, self.idx_cols_unique = preprocess(self.X)

            # preprocess A and B
            self.A = self.A[self.idx_rows_unique, :]
            self.B = self.B[:, self.idx_cols_unique]

            if self.idx_zero_row is not None:
                self.A = np.delete(self.A, self.idx_zero_row, axis=0)
            if self.idx_zero_col is not None:
                self.B = np.delete(self.B, self.idx_zero_col, axis=1)

    def post_process_output_matrices(self):
        """
        Place back duplicate and zero rows/columns so it corresponds to original input matrix
        """

        if self.idx_rows_reconstruct is None or \
                self.idx_cols_reconstruct is None or  \
                self.idx_rows_unique is None or \
                self.idx_cols_unique is None:
            raise ValueError(
                'Problem has not been preprocessed so it cannot be post-processed,\n use .preprocess_input_matrices() to preprocess')

        self.A, self.B = post_process_factorisation(self.A, self.B, self.idx_rows_reconstruct, self.idx_zero_row, self.idx_cols_reconstruct, self.idx_zero_col)

        self.X = un_preprocess(self.X, self.idx_rows_reconstruct, self.idx_zero_row, self.idx_cols_reconstruct, self.idx_zero_col)

        # if matrices are set back to the original dimension, update the preprocessing paramaters to default too
        self.row_weights = np.ones(self.X.shape[0], dtype=int)
        self.col_weights = np.ones(self.X.shape[1], dtype=int)

        self.idx_rows_reconstruct, self.idx_cols_reconstruct, \
        self.idx_zero_row, self.idx_zero_col, \
        self.idx_rows_unique, self.idx_cols_unique = [None] * 6

    def CIP_set_cplex_model(self):
        """
        Set up the compact integer program CIP in cplex
        """

        n, m = self.X.shape

        idx0 = np.transpose(np.where(self.X == 0))
        idx1 = np.transpose(np.where(self.X == 1))
        idxs = np.concatenate((idx0, idx1))

        self.CIP_model = cplex.Cplex()
        self.CIP_model.objective.set_sense(self.CIP_model.objective.sense.minimize)

        z0_names = ["z(%s,%s)" % (i, j) for (i, j) in idx0]
        self.CIP_model.variables.add(lb=[0] * len(idx0),
                                     ub=[1] * len(idx0),
                                     types=[self.CIP_model.variables.type.continuous] * len(idx0),
                                     names=z0_names)
        z1_names = ["z(%s,%s)" % (i, j) for (i, j) in idx1]
        self.CIP_model.variables.add(lb=[0] * len(idx1),
                                     ub=[1] * len(idx1),
                                     types=[self.CIP_model.variables.type.continuous] * len(idx1),
                                     names=z1_names)
        z_names = z0_names + z1_names
        self.CIP_model.z_names = z_names

        # a_i,l \in {0,1}
        a_names = ["a(%s,%s)" % (i, l) for i in range(n) for l in range(self.k)]
        self.CIP_model.variables.add(lb=[0] * n * self.k,
                                     ub=[1] * n * self.k,
                                     types=[self.CIP_model.variables.type.binary] * n * self.k,
                                     names=a_names)
        self.CIP_model.a_names = a_names

        # b_l,j \in {0,1}
        b_names = ["b(%s,%s)" % (l, j) for l in range(self.k) for j in range(m)]
        self.CIP_model.variables.add(lb=[0] * self.k * m,
                                     ub=[1] * self.k * m,
                                     types=[self.CIP_model.variables.type.binary] * self.k * m,
                                     names=b_names)
        self.CIP_model.b_names = b_names

        # y_i,l,j \in [0,1]
        y_names = ["y(%s,%s,%s)" % (i, l, j) for i in range(n) for l in range(self.k) for j in range(m)]
        self.CIP_model.variables.add(lb=[0] * n * self.k * m,
                                     ub=[1] * n * self.k * m,
                                     types=[self.CIP_model.variables.type.continuous] * n * self.k * m,
                                     names=y_names)
        self.CIP_model.y_names = y_names

        # objective

        self.CIP_model.objective.set_linear(
            [("z(%s,%s)" % (i, j), float(self.row_weights[i] * self.col_weights[j]))
             for (i, j) in idx0
             ]
        )

        self.CIP_model.objective.set_linear(
            [("z(%s,%s)" % (i, j), -1 * float(self.row_weights[i] * self.col_weights[j]))
             for (i, j) in idx1
             ]
        )

        self.CIP_model.objective.set_offset(float(sum(self.row_weights[i] * self.col_weights[j] for (i, j) in idx1)))

        # Boolean Matrix Product constraints

        # y_i,l,j <= z_i,j          when x_ij = 0
        self.CIP_model.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=["y(%s,%s,%s)" % (i, l, j), "z(%s,%s)" % (i, j)],
                                 val=[1, -1])
                for l in range(self.k) for (i, j) in idx0
            ],
            rhs=[0] * self.k * len(idx0),
            senses=["L"] * self.k * len(idx0),
            names=[
                "y(%s,%s,%s) <= z(%s,%s)" % (i, l, j, i, j) for l in range(self.k) for (i, j) in idx0
            ]
        )

        # z_i,j <= sum_l y_i,l,j        when x_ij = 1
        self.CIP_model.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=["z(%s,%s)" % (i, j)] + ["y(%s,%s,%s)" % (i, l, j) for l in range(self.k)],
                                 val=[1] + [-1 for l in range(self.k)])
                for (i, j) in idx1
            ],
            rhs=[0] * len(idx1),
            senses=["L"] * len(idx1),
            names=["z(%s,%s) <= sum_k y(%s,l,%s)" % (i, j, i, j) for (i, j) in idx1]
        )

        # McCormick constraints

        # y_i,l,j <= a_i,l
        self.CIP_model.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=["y(%s,%s,%s)" % (i, l, j), "a(%s,%s)" % (i, l)],
                                 val=[1, -1])
                for l in range(self.k) for (i, j) in idxs
            ],
            rhs=[0] * self.k * len(idxs),
            senses=["L"] * self.k * len(idxs),
            names=["y(%s,%s,%s) <=  a(%s,%s)" % (i, l, j, i, l) for l in range(self.k) for (i, j) in idxs]
        )

        # y_i,l,j <= b_l,j
        self.CIP_model.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=["y(%s,%s,%s)" % (i, l, j), "b(%s,%s)" % (l, j)],
                                 val=[1, -1])
                for l in range(self.k) for (i, j) in idxs
            ],
            rhs=[0] * self.k * len(idxs),
            senses=["L"] * self.k * len(idxs),
            names=["y(%s,%s,%s) <=  b(%s,%s)" % (i, l, j, l, j) for l in range(self.k) for (i, j) in idxs]
        )

        # a_i,l + b_l,j - 1 <= y_i,l,j
        self.CIP_model.linear_constraints.add(
            lin_expr=[
                cplex.SparsePair(ind=["a(%s,%s)" % (i, l), "b(%s,%s)" % (l, j), "y(%s,%s,%s)" % (i, l, j)],
                                 val=[1, 1, -1])
                for l in range(self.k) for (i, j) in idxs
            ],
            rhs=[1] * self.k * len(idxs),
            senses=["L"] * self.k * len(idxs),
            names=["a(%s,%s) + b(%s,%s) - 1 <= y(%s,%s,%s)" % (i, l, l, j, i, l, j)
                   for l in range(self.k) for (i, j) in idxs
                   ]
        )

    def CIP_solve(self, max_time=1000, symmetry_break = 5, display=2):
        """
        Solve the compact integer program CIP for rank-k binary matrix factorisation
        To obtain factorisation get self.A, self.B

        Parameters
        ----------
        max_time :                  maximum time in seconds, default=1000
        symmetry_break :            int in [1,5], default=5, most aggressive symmetry breaking in cplex,
                                    to set parameters.preprocessing.symmetry, see cplex documentation
        symmetry_break :            int, default=2,
                                    to set parameters.mip.display, see cplex documentation

        """

        self.CIP_set_cplex_model()

        self.CIP_model.parameters.mip.display.set(display)

        if (isinstance(max_time , int) or isinstance(max_time , float)) and max_time > 0 :
            # 'time limit in seconds '
            self.CIP_model.parameters.timelimit.set(max_time)

        if symmetry_break in range(-1, 6):
            # 'indicator for symmetric reductions  :\n  -1 = automatic\n  0 = off\n  1-5 = increasing aggressive levels'
            self.CIP_model.parameters.preprocessing.symmetry.set(symmetry_break)

        if self.A.shape[1] > 0:
            if self.A.shape[1] < self.k:
                self.A = np.pad(self.A, ((0, 0), (0, self.k - self.A.shape[1])), 'constant', constant_values=0)
                self.B = np.pad(self.B, ((0, self.k - self.B.shape[0]), (0, 0)), 'constant', constant_values=0)
            elif self.A.shape[1] > self.k:
                self.A = self.A[:, range(self.k)]
                self.B = self.B[range(self.k), :]

            Z = boolean_matrix_product(self.A, self.B)

            Y = np.array([[[self.A[i, l] * self.B[l, j] for j in range(self.X.shape[1])] for l in range(self.k)] for i in range(self.X.shape[0])])

            z_values = np.concatenate([Z[self.X == 0], Z[self.X == 1]]).tolist()
            a_vales = self.A.flatten().tolist()
            b_values = self.B.flatten().tolist()
            y_values = Y.flatten().tolist()

            self.CIP_model.MIP_starts.add(cplex.SparsePair(self.CIP_model.z_names +
                                                           self.CIP_model.a_names +
                                                           self.CIP_model.b_names +
                                                           self.CIP_model.y_names,
                                                           z_values +
                                                           a_vales +
                                                           b_values +
                                                           y_values),
                                          self.CIP_model.MIP_starts.effort_level.auto,
                                          'CIP_A_B_init')

        # Solve CPLEX IP
        self.CIP_model.solve()

        print(self.CIP_model.solution.status[self.CIP_model.solution.get_status()])

        A = np.array(self.CIP_model.solution.get_values(self.CIP_model.a_names)).reshape(self.X.shape[0], self.k)
        B = np.array(self.CIP_model.solution.get_values(self.CIP_model.b_names)).reshape(self.k, self.X.shape[1])

        A[abs(A) < EPS] = 0
        A[abs(A - 1) < EPS] = 1
        B[abs(B) < EPS] = 0
        B[abs(B - 1) < EPS] = 1

        self.A = A.astype(int)
        self.B = B.astype(int)




if __name__ == "__main__":

    # binary matrix generation, preprocessing, post-processing
    XX = random_binary_matrix(20, 10, 10, 0.5, 0, 0)
    print('The original dimension is \t\t\t', XX.shape)

    XX_out, row_weights, col_weights, idx_rows_reconstruct, idx_cols_reconstruct,\
    idx_zero_row, idx_zero_col, idx_rows_unique, idx_cols_unique = preprocess(XX)
    print('The preprocessed dimension is \t\t', XX_out.shape)

    YY = un_preprocess(XX_out, idx_rows_reconstruct, idx_zero_row, idx_cols_reconstruct, idx_zero_col)
    print('The un-preprocessed dimension is \t', YY.shape)
    print('The original and output matrix are equal ', np.array_equal(XX, YY))

    kk = 3

    # k-Greedy heuristic for rank-k binary matrix factorisation
    A_greedy, B_greedy = BMF_k_greedy_heur(XX, kk)

    # using column generation class to get rank-k binary matrix factorisation
    cg = BMF_via_CG(XX, kk, A_init=A_greedy, B_init=B_greedy)
    cg.preprocess_input_matrices()
    cg.MLP_solve(max_time=20, display=True, log=True)
    #print(cg.MLP_log)

    # solve MIP(1)
    cg.MIP_solve(max_time=10)
    cg.post_process_output_matrices()
    A_MIP1 = cg.A
    B_MIP1 = cg.B

    # solve MIP with frobenius objective
    cg.preprocess_input_matrices()
    cg.MIP_solve(objective = 'frobenius', max_time=10)
    cg.post_process_output_matrices()
    A_MIP_F = cg.A
    B_MIP_F = cg.B


    # using compact integer program class to get rank-k binary matrix factorisation
    cip = BMF_via_compact_IP(XX, kk, A_init=A_greedy, B_init=B_greedy)
    cip.preprocess_input_matrices()
    cip.CIP_solve(max_time=20)
    cip.post_process_output_matrices()
    A_CIP = cip.A
    B_CIP = cip.B

    print('\n\nThe rank-%s BMF via k-Greedy has error \t\t\t\t\t\t%s' %(kk, np.sum(np.abs(XX - boolean_matrix_product(A_greedy, B_greedy)))))

    print('\nMLP(1) generated %s rank-1 binary matrices and dual bound\t%s' % (cg.MLP_A.shape[1]-kk, cg.best_dual_bound))

    print('\n\nThe rank-%s BMF via MIP(1) has error \t\t\t\t\t\t%s' %(kk, np.sum(np.abs(XX - boolean_matrix_product(A_MIP1, B_MIP1)))))

    print('\n\nThe rank-%s BMF via MIP_F has error \t\t\t\t\t\t\t%s' %(kk, np.sum(np.abs(XX - boolean_matrix_product(A_MIP_F, B_MIP_F)))))

    print('\n\nThe rank-%s BMF via CIP has error \t\t\t\t\t\t\t%s' %(kk, np.sum(np.abs(XX - boolean_matrix_product(A_CIP, B_CIP)))))

