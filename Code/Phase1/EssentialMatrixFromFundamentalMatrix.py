import numpy as np


def compute_essential_matrix(F,K):
    E = K.T @  F @ K
    print(f"Essential Matrix E before SVD:{E}")
    U,S,Vt = np.linalg.svd(E)
    S_corrected = np.array([1,1,0])
    E_corrected = U @ np.diag(S_corrected)@ Vt
    return E_corrected