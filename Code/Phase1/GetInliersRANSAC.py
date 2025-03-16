import cv2 
import numpy as np
from EstimateFundamentalMatrix import *



num_iterations = 2000
threshold = 0.008
num_inliers = 0 


def F_error(coord1, coord2, F):
    x1 = np.array([coord1[0], coord1[1], 1.0])
    x2 = np.array([coord2[0], coord2[1], 1.0])
    error = x2 @ (F @ x1)   # numpy is handling transpose by using  @ opperator iit is a doot product operator which converts the x2 to a vector 
    
    
    return abs(error)

def getInliners(coord1,coord2,num_iterations, threshold):
        N = coord1.shape[0]
        best_inlier_count = 0
        best_inlier_idx = []
        best_F=None
        
        for _ in range(num_iterations):
            sample_indices = np.random.choice(N,8,replace=False)
            pts1_sample = coord1[sample_indices]
            pts2_sample = coord2[sample_indices]
            # print(f"point 1 sample shape{ pts1_sample.shape}")
            # print(f"point 2 sample shape {pts2_sample.shape}")
            F_current = estimate_fundamental_matrix(pts1_sample,pts2_sample)
        
            inliers_idx = []
            for j in range(N):
                error = F_error(coord1[j],coord2[j],F_current)
                if error < threshold:
                    inliers_idx.append(j)
                    
            inliers_count = len(inliers_idx)
            if inliers_count> best_inlier_count:
                best_inlier_count = inliers_count
                best_inlier_idx = inliers_idx
                best_F = F_current
                # print(f"Best F is : ")
        return best_F , best_inlier_count
        
        

        