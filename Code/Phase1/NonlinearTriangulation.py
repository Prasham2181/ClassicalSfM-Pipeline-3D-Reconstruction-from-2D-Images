import numpy as np 
from scipy.optimize import least_squares

"""Linear triangulation method minimizes algebraic errorr whic is sensitive to noise in unpredictible ways

also this algebric error is not directly related to the geometric reprojection error whic is a 
euclidian distance between the bserved 2D point and its projected 3D point 

While its advantage is that its quick but the problem is with the accuracy 


linear triangullation works on x ~ pX linear equation and then we solve AX = 0 using svd and this method minimizes the algebric error


"""



def non_linearTriangulation(K, R1, C1, R2, C2, X0, x1_obs, x2_obs):
    """
    Non-linear triangulation to refine 3D point position by minimizing reprojection error.
    
    Args:
        K: Camera intrinsic matrix
        R1, C1: Rotation and camera center for first camera
        R2, C2: Rotation and camera center for second camera
        X0: Initial 3D point estimate from linear triangulation
        x1_obs, x2_obs: Observed 2D points in the two images
        
    Returns:
        X_refined: Refined 3D point
    """
    P1 = projection_matrix_p(K, R1, C1)
    P2 = projection_matrix_p(K, R2, C2)
    
    def error(X):
        # Compute reprojection error for both cameras
        X_h = np.append(X, 1)  # Convert to homogeneous
        
        # Project to camera 1
        x1_proj_h = P1 @ X_h
        x1_proj = x1_proj_h[:2] / x1_proj_h[2]
        err1 = x1_obs - x1_proj
        
        # Project to camera 2
        x2_proj_h = P2 @ X_h
        x2_proj = x2_proj_h[:2] / x2_proj_h[2]
        err2 = x2_obs - x2_proj
        
        return np.hstack((err1, err2))
    
    # Use Levenberg-Marquardt algorithm with better parameters
    result = least_squares(
        error, 
        X0, 
        method='lm',
        ftol=1e-4,
        xtol=1e-4,
        max_nfev=50  # Limit iterations to avoid excessive computation
    )
    
    X_refined = result.x
    
    # Check if optimization improved the result
    initial_error = np.linalg.norm(error(X0))
    final_error = np.linalg.norm(error(X_refined))
    
    # If optimization failed to improve, keep the original estimate
    if final_error >= initial_error:
        return X0
        
    return X_refined

# def reprojection_error(X , P , x_obs):
#     if X.shape[0] == 3:
#         X_h = np.hstack((X,1))
#     else:
#         X_h = X
    
#     x_proj_h = P @ X_h 
#     # print("X Project_h" , x_proj_h)
#     x_proj = x_proj_h[ :2] / x_proj_h[2]
#     error = x_obs - x_proj
    
#     return error 

def reprojection_error(X, P, x_obs):
    """
    Calculate reprojection error between projected 3D point and observed 2D point.
    
    Args:
        X: 3D point (3, ) or (4, ) if homogeneous
        P: Projection matrix (3x4)
        x_obs: Observed 2D point (2, )
        
    Returns:
        error: Reprojection error vector (2, )
    """
    # Convert to homogeneous if needed
    if X.shape[0] == 3:
        X_h = np.append(X, 1)
    else:
        X_h = X
    
    # Project 3D point to image
    x_proj_h = P @ X_h
    
    # Handle potential division by zero
    if abs(x_proj_h[2]) < 1e-10:
        return np.array([float('inf'), float('inf')])
    
    # Convert to inhomogeneous coordinates
    x_proj = x_proj_h[:2] / x_proj_h[2]
    
    # Calculate reprojection error
    error = x_obs - x_proj
    
    return error

def projection_matrix_p(K,R,C):
    #  for calibtated camera projection matrix P = K [R | -R C]
        C = C.reshape(3,1)
        P = K@ np.hstack((R, -R@C)) 
        # print(f"Projection Matrix Size  : {P.shape}")
        return P 
    
def filter_points_by_error(X, P1, P2, pts1, pts2, threshold=3.0):
    """Filter 3D points by reprojection error."""
    N = X.shape[0]
    good_points = []
    
    for i in range(N):
        # Project the point into both cameras
        X_h = np.append(X[i], 1)
        
        # Project to image 1
        x1_proj = P1 @ X_h
        x1_proj = x1_proj[:2] / x1_proj[2]
        
        # Project to image 2
        x2_proj = P2 @ X_h
        x2_proj = x2_proj[:2] / x2_proj[2]
        
        # Calculate reprojection errors
        err1 = np.linalg.norm(x1_proj - pts1[i])
        err2 = np.linalg.norm(x2_proj - pts2[i])
        
        # Keep point if errors are small in both views
        if err1 < threshold and err2 < threshold:
            good_points.append(i)
    
    return good_points 
    
    
    