import numpy as np 




def linearTriangulation(K, C1, R1, C2, R2, x1, x2):
    """
    Linear triangulation method using Direct Linear Transform (DLT).
    
    Args:
        K: Camera intrinsic matrix
        C1, R1: Camera center and rotation for first camera
        C2, R2: Camera center and rotation for second camera
        x1, x2: 2D points in both images
        
    Returns:
        X: Triangulated 3D points
        inlier_mask: Boolean mask of points with positive depth
    """
    N = x1.shape[0]
    X = np.zeros((N, 3))
    depths = np.zeros((N, 2))  # Store depths for both cameras
    
    P1 = K @ np.hstack((R1, -R1 @ C1.reshape(3, 1)))
    P2 = K @ np.hstack((R2, -R2 @ C2.reshape(3, 1)))
    
    for i in range(N):
        u1, v1 = x1[i]
        u2, v2 = x2[i]
        
        # Create the A matrix for DLT
        A = np.zeros((4, 4))
        A[0, :] = u1 * P1[2, :] - P1[0, :]
        A[1, :] = v1 * P1[2, :] - P1[1, :]
        A[2, :] = u2 * P2[2, :] - P2[0, :]
        A[3, :] = v2 * P2[2, :] - P2[1, :]
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X_h = Vt[-1]
        X_h = X_h / X_h[-1]
        X[i, :] = X_h[:3]
        
        # Calculate depths in both cameras
        # Transform point to camera coordinates
        X_c1 = R1 @ X[i, :] + (-R1 @ C1)
        X_c2 = R2 @ X[i, :] + (-R2 @ C2)
        
        # Store Z-coordinate (depth)
        depths[i, 0] = X_c1[2]
        depths[i, 1] = X_c2[2]
    
    # Create inlier mask for points with positive depth in both cameras
    inlier_mask = (depths[:, 0] > 0) & (depths[:, 1] > 0)
    
    return X, inlier_mask