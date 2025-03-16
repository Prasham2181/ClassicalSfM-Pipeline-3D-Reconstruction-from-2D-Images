import numpy as np
from LinearPnP import linearPnP


import numpy as np
from LinearPnP import linearPnP

def pnp_ransac(pts_3D, pts_2D, K, num_iterations=1000, reproj_threshold=3.0):
   
    N = pts_3D.shape[0]
    if N < 6:
        print(f"Warning: Not enough points for PnP RANSAC ({N} < 6)")
        R, C, _ = linearPnP(pts_3D, pts_2D, K)
        if R is None:
            return None, None, []
        return R, C, list(range(N))
    
    R, C, inlier_mask = linearPnP(pts_3D, pts_2D, K, ransac_threshold=reproj_threshold, iterations=num_iterations)
    if R is None:
        return None, None, []
    
    inlier_indices = np.where(inlier_mask)[0].tolist()
    return R, C, inlier_indices

def get_pnp_Ransac(pts_3D, pts_2D, K, num_iterations=1000, reproj_threshold=3.0):
    return pnp_ransac(pts_3D, pts_2D, K, num_iterations, reproj_threshold)
