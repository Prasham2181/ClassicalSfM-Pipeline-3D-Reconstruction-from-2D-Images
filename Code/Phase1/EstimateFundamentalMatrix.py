import numpy as np
import glob



# to estimate the fundamental matix i refered this website:

# https://sites.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html



def normalize_points(points):

    N = points.shape[0]
    centroid = np.mean(points, axis=0)  # [mean_u, mean_v]
    
    points_centered = points - centroid
    
    mean_dist = np.mean(np.sqrt(np.sum(points_centered**2, axis=1)))
    
    scale = np.sqrt(2) / mean_dist
    
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    #    here t is the normalized matrix and points_ normalized are the stored at points_normalized
    points_hom = np.hstack((points, np.ones((N, 1))))
    
    points_normalized = (T @ points_hom.T).T
    
    return points_normalized, T


def build_equation_matrix(pts1_norm,pts2_norm):
    N = pts1_norm.shape[0]
    A = np.zeros((N,9))
    
    for i in range(N):
        u1,v1, _ = pts1_norm[i]
        u2,v2, _ = pts2_norm[i]
        
        A[i] = [ u2*u1, u2*v1 , u2 , v2*u1, v2*v1 , v2, u1 , v1 , 1]
        
        
    return A 
    
    #  we have the equation Af = 0 with A computed using normalized coordinates u and v so 
    # now we do svd and enforse rank 2 
    
    
def compute_F_from_A(A):
    # A matrix is Nx9
    # and F has to be 3x3 and thenn normalized to rank 2 
    
    U, S , Vt = np.linalg.svd(A)
    f = Vt[-1]
    F_norm = f.reshape(3,3)
    #  enforcing rank 2 
    
    
    U_f,D_f, Vt_f = np.linalg.svd(F_norm)
    D_f [2] = 0 # we set the last element of the 3x3 mmatrix to zero so that we get a rank 2
    
    F_norm_rank2 =  U_f @ np.diag(D_f)@ Vt_f
    
    return F_norm_rank2

#  now as we want the fundamental matrix to work on the acutal values of the image we need to denorm it to the original pixel values 


def denrom_F(F_norm, T1 , T2):
    
    
    F = T2.T @ F_norm@T1
    F = F / np.linalg.norm(F)
    
    return F


def estimate_fundamental_matrix(pts1 , pts2 ):
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)
    
    
    A = build_equation_matrix(pts1_norm,pts2_norm)
    F_norm = compute_F_from_A(A)
    
    F = denrom_F(F_norm , T1 , T2)
    
    return F

    
    
    