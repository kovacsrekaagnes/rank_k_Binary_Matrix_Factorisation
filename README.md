
# Binary Matrix Factorisation and Completion via Integer Programming



## Prerequisites

Python 3.6.8 with the following libraries installed
- numpy 1.15.4
- pandas 0.23.4
- networkx 2.4
- scipy 1.1.0
- math 
- cplex 12.8.0.0 (free for academic use but needs license)


## DEMO


```
from BMF import *
```

####Synthetic matrix

Create a 20 x 10 binary matrix with Boolean rank at most 10, with 50% zeros, 0% noise and random seed 0:
```
    X = random_binary_matrix(20, 10, 10, 0.5, 0, 0)
```
If want to add some missing entries set some entries to np.nan, which we use as the value to denote missing entries.

#### k-Greedy heuristic
Use the k-Greedy heuristic for a rank-3 binary matrix factorisation and compute the factorisation error:
```
    A_greedy, B_greedy = BMF_k_greedy_heur(X, 3)
    print(np.sum(np.abs(X - boolean_matrix_product(A_greedy, B_greedy))))
```
#### Column generation
Use the BMF_via_CG class to get a rank-3 BMF.
Give the k-Greedy heuristic solution as a warm start, preprocess X to obtain a possibly smaller but equivalent problem and set a time limit of 20 seconds on the column generation procedure:
```
    cg = BMF_via_CG(X, 3, A_init=A_greedy, B_init=B_greedy)
    cg.preprocess_input_matrices()
    cg.MLP_solve(max_time=20)
```
After the rank-1 binary matrices are generated, 
solve the default master IP model MIP(1) with a time limit of 10 seconds to choose 3 rank-1 binary matrices:
```
    cg.MIP_solve(max_time=10)
```
Extract the rank-3 binary matrix factorisation after post processing, 
so the factorisation is in the original dimension and compute the factorisation error in the original squared Frobenius norm objective:
```
    cg.post_process_output_matrices()
    A_MIP1 = cg.A
    B_MIP1 = cg.B
    print(np.sum(np.abs(X - boolean_matrix_product(A_MIP1, B_MIP1))))
```

We could have also solved model MIP_F to to choose 3 rank-1 binary matrices:
```
    cg.preprocess_input_matrices() # if problem is not already preprocessed
    cg.MIP_solve(objective = 'frobenius', max_time=10)
    cg.post_process_output_matrices()
    A_MIP_F = cg.A
    B_MIP_F = cg.B
    print(np.sum(np.abs(X - boolean_matrix_product(A_F, B_F))))
```

#### Compact integer program

Use the compact integer program to get a rank-3 binary matrix factorisation
with a warm start of the k-Greedy heuristic solution and a time limit of 20 seconds: 

```
    cip = BMF_via_compact_IP(X, k, A_init=A_greedy, B_init=B_greedy)
    cip.preprocess_input_matrices()
    cip.CIP_solve(max_time=20)
    cip.post_process_output_matrices()
    A_CIP = cip.A
    B_CIP = cip.B
    print(np.sum(np.abs(X - boolean_matrix_product(A_CIP, B_CIP))))
```



#### Real data

Raw data obtained from:
- apb http://moreno.ss.uci.edu/data.html#books
- audio http://archive.ics.uci.edu/ml/datasets/audiology+(standardized)
- heart https://archive.ics.uci.edu/ml/datasets/spect+heart
- hepatitis https://archive.ics.uci.edu/ml/datasets/Hepatitis
- lymp https://archive.ics.uci.edu/ml/datasets/Lymphography
- tumor https://archive.ics.uci.edu/ml/datasets/Primary+Tumor
- votes https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
- zoo http://archive.ics.uci.edu/ml/datasets/zoo 

To see how data is binarised look at:
```
binarise_data.py
```
