import numpy as np
import cv2

def homogenize_points(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def projection_matrix_p(K, R, C):
    """
    Construct the projection matrix from camera intrinsics K,
    rotation R and camera center C.
    """
    C = C.reshape(3, 1)
    P = K @ np.hstack((R, -R @ C))
    return P

def linearPnP(pts_3D, pts_2D, K, ransac_threshold=3.0, iterations=5000):
    """
    Estimate the camera pose using PnP with RANSAC (using OpenCV).
    
    Args:
        pts_3D: Array of 3D points (N x 3).
        pts_2D: Array of corresponding 2D points (N x 2).
        K: Camera intrinsic matrix.
        ransac_threshold: Reprojection error threshold.
        iterations: Number of RANSAC iterations.
        
    Returns:
        R: Rotation matrix (3x3).
        C: Camera center (3,) computed as -R^T * tvec.
        inlier_mask: Boolean array indicating inlier correspondences.
    """
    pts_3D = pts_3D.astype(np.float32)
    pts_2D = pts_2D.astype(np.float32)
    
    if pts_3D.shape[0] < 6:
        print(f"Warning: Not enough points for PnP ({pts_3D.shape[0]} < 6)")
        return None, None, None

    # First try P3P (requires at least 4 points)
    success = False
    try:
        if pts_3D.shape[0] >= 4:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3D, pts_2D, K, None,
                iterationsCount=iterations,
                reprojectionError=ransac_threshold,
                flags=cv2.SOLVEPNP_P3P
            )
    except cv2.error:
        success = False

    # If P3P fails, try the default method
    if not success:
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3D, pts_2D, K, None,
                iterationsCount=iterations,
                reprojectionError=ransac_threshold
            )
        except cv2.error:
            print("PnP RANSAC failed")
            return None, None, None
    
    if not success:
        print("PnP RANSAC failed to find a good solution")
        return None, None, None

    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Compute the camera center using the relation t = -R * C,
    # so that C = -R^T * tvec.
    C = -R.T @ tvec

    # Build the inlier mask
    if inliers is None:
        inlier_mask = np.ones(pts_3D.shape[0], dtype=bool)
    else:
        inlier_mask = np.zeros(pts_3D.shape[0], dtype=bool)
        inlier_mask[inliers.ravel()] = True

    print(f"PnP found {np.sum(inlier_mask)} inliers out of {pts_3D.shape[0]} points")
    
    return R, C.flatten(), inlier_mask
