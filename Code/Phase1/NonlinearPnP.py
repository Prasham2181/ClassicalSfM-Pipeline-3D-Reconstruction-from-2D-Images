import numpy as np 
from scipy.optimize import least_squares
from scipy.optimize import leastsq
import cv2
#  linear pnp is used to yreduce the Algebric error while non linear pnp is use ro redice geometric error which is a better
#  way for projection 
def quat_to_rot(q):
    """
    Converts a unit quaternion q = [q0, q1, q2, q3] into a rotation matrix R.
    Assumes q is normalized.
    """
    q0, q1, q2, q3 = q
    R = np.array([
        [q0**2 + q1**2 - q2**2 - q3**2, 2*(q1*q2 - q0*q3),         2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),         q0**2 - q1**2 + q2**2 - q3**2, 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2),         2*(q2*q3 + q0*q1),         q0**2 - q1**2 - q2**2 + q3**2]
    ])
    return R

def project_points(pts_3D, R, t, K):
  
    P = K @ np.hstack((R, t.reshape(3, 1)))  # Compute projection matrix.
    pts_3D_h = np.hstack((pts_3D, np.ones((pts_3D.shape[0], 1)))).T  # (4, N)
    pts_proj_h = P @ pts_3D_h  # (3, N)
    pts_proj = (pts_proj_h / pts_proj_h[2, :]).T  # Normalize homogeneous coordinates.
    return pts_proj[:, :2]

def rot_to_quat(R):
    
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q0 = 0.25 / s
        q1 = (R[2,1] - R[1,2]) * s
        q2 = (R[0,2] - R[2,0]) * s
        q3 = (R[1,0] - R[0,1]) * s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            q0 = (R[2,1] - R[1,2]) / s
            q1 = 0.25 * s
            q2 = (R[0,1] + R[1,0]) / s
            q3 = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            q0 = (R[0,2] - R[2,0]) / s
            q1 = (R[0,1] + R[1,0]) / s
            q2 = 0.25 * s
            q3 = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            q0 = (R[1,0] - R[0,1]) / s
            q1 = (R[0,2] + R[2,0]) / s
            q2 = (R[1,2] + R[2,1]) / s
            q3 = 0.25 * s
    q = np.array([q0, q1, q2, q3])
    return q / np.linalg.norm(q)

def cost_function(params, pts_3D, pts_2D, K):
    
    C = params[:3]
    q = params[3:7]
    q = q / np.linalg.norm(q)  # Ensure unit quaternion.
    R = quat_to_rot(q)
    t = -R @ C  # t = -R * C.
    pts_proj = project_points(pts_3D, R, t, K)
    residuals = (pts_proj - pts_2D).ravel()
    return residuals

# def nonlinearPnP(pts_3D, pts_2D, K, initial_C, initial_R):
#     """
#     Refines the camera pose (C, R) given 2D-3D correspondences and an initial guess.
#     pts_3D: (N, 3) array of 3D points.
#     pts_2D: (N, 2) array of corresponding 2D points in pixel coordinates.
#     K: Intrinsic matrix.
#     initial_C: Initial camera center (3,).
#     initial_R: Initial rotation matrix (3x3).
#     Returns refined camera center and rotation.
#     """
#     initial_q = rot_to_quat(initial_R)
#     initial_params = np.hstack([initial_C, initial_q])
    
#     result = least_squares(cost_function, initial_params, args=(pts_3D, pts_2D, K))
#     refined_params = result.x
#     refined_C = refined_params[:3]
#     refined_q = refined_params[3:7] / np.linalg.norm(refined_params[3:7])
#     refined_R = quat_to_rot(refined_q)
#     return refined_C, refined_R

def nonlinearPnP(pts_3D, pts_2D, K, C_init, R_init, max_iter=100, tol=1e-8):
    """
    Non-linear refinement of camera pose to minimize reprojection error.
    
    Args:
        pts_3D: 3D points (N x 3)
        pts_2D: Corresponding 2D points (N x 2)
        K: Camera intrinsic matrix
        C_init: Initial camera center
        R_init: Initial rotation matrix
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        R: Refined rotation matrix
        C: Refined camera center
    """
    from scipy.optimize import least_squares
    
    # Convert rotation matrix to Rodrigues vector
    rvec_init, _ = cv2.Rodrigues(R_init)
    # Convert camera center to translation vector
    tvec_init = -R_init @ C_init
    
    # Flatten parameters for optimization
    params_init = np.hstack([rvec_init.flatten(), tvec_init.flatten()])
    
    # Define the cost function to minimize
    def cost_function(params):
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:].reshape(3, 1)
        
        # Project all points
        projected_pts, _ = cv2.projectPoints(pts_3D, rvec, tvec, K, None)
        projected_pts = projected_pts.reshape(-1, 2)
        
        # Calculate reprojection errors
        errors = projected_pts - pts_2D
        
        return errors.flatten()
    
    # Run optimization
    result = least_squares(
        cost_function,
        params_init,
        method='lm',
        max_nfev=max_iter,
        ftol=tol,
        xtol=tol
    )
    
    # Extract optimized parameters
    params_opt = result.x
    rvec_opt = params_opt[:3].reshape(3, 1)
    tvec_opt = params_opt[3:].reshape(3, 1)
    
    # Convert back to rotation matrix and camera center
    R_opt, _ = cv2.Rodrigues(rvec_opt)
    C_opt = -R_opt.T @ tvec_opt
    
    return C_opt.flatten(), R_opt
