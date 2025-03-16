import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import cv2

def bundle_adjustment(cameras, points_3d, points_2d, camera_indices, point_indices, K):

    # Parameters
    n_cameras = len(cameras)
    n_points = len(points_3d)
    n_observations = len(points_2d)
    
    # Calculate parameter vector size
    n_params = 6 * n_cameras + 3 * n_points
    
    # Initialize parameter vector
    params = np.zeros(n_params)
    
    # Fill in initial camera parameters
    for i, (R, C) in enumerate(cameras):
        # Convert R to Rodrigues rotation vector
        rvec, _ = cv2.Rodrigues(R)
        # Translation vector t = -R*C
        tvec = -R @ C
        # Add to parameter vector
        params[6*i:6*i+3] = rvec.flatten()
        params[6*i+3:6*i+6] = tvec.flatten()
    
    # Fill in initial point parameters
    for i, point in enumerate(points_3d):
        params[6*n_cameras+3*i:6*n_cameras+3*i+3] = point
    
    # Create sparsity matrix for Jacobian
    A = lil_matrix((2 * n_observations, n_params), dtype=int)
    
    # Fill in sparsity pattern
    for i in range(n_observations):
        # Camera parameters affect this observation
        cam_idx = camera_indices[i]
        for j in range(6):
            A[2*i, 6*cam_idx+j] = 1
            A[2*i+1, 6*cam_idx+j] = 1
        
        # Point parameters affect this observation
        pt_idx = point_indices[i]
        for j in range(3):
            A[2*i, 6*n_cameras+3*pt_idx+j] = 1
            A[2*i+1, 6*n_cameras+3*pt_idx+j] = 1
    
    # Convert to CSR format for efficiency
    A = A.tocsr()
    
    # Define cost function for optimization
    def cost_function(params):
        """Calculate reprojection error for each observation."""
        # Extract camera parameters
        camera_params = params[:6*n_cameras].reshape((n_cameras, 6))
        # Extract point parameters
        point_params = params[6*n_cameras:].reshape((n_points, 3))
        
        # Calculate reprojection errors
        errors = np.zeros(2 * n_observations)
        
        for i in range(n_observations):
            cam_idx = camera_indices[i]
            pt_idx = point_indices[i]
            
            # Get camera parameters
            rvec = camera_params[cam_idx, :3].reshape(3, 1)
            tvec = camera_params[cam_idx, 3:6].reshape(3, 1)
            
            # Get 3D point
            point_3d = point_params[pt_idx].reshape(1, 3)
            
            # Project 3D point to image plane
            projected_pt, _ = cv2.projectPoints(point_3d, rvec, tvec, K, None)
            projected_pt = projected_pt.flatten()
            
            # Calculate reprojection error
            errors[2*i:2*i+2] = projected_pt - points_2d[i]
        
        return errors
    
    # Run optimization
    print("Starting bundle adjustment with", n_cameras, "cameras and", n_points, "points...")
    result = least_squares(
        cost_function, 
        params, 
        jac_sparsity=A, 
        verbose=2, 
        x_scale='jac', 
        ftol=1e-6, 
        method='trf',
        max_nfev=200,# Limit iterations for large problems
        loss='huber'
    )
    
    # Extract refined parameters
    refined_params = result.x
    camera_params = refined_params[:6*n_cameras].reshape((n_cameras, 6))
    point_params = refined_params[6*n_cameras:].reshape((n_points, 3))
    
    # Convert back to camera poses
    refined_cameras = []
    for i in range(n_cameras):
        rvec = camera_params[i, :3].reshape(3, 1)
        tvec = camera_params[i, 3:6].reshape(3, 1)
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        # Calculate camera center
        C = -R.T @ tvec
        
        refined_cameras.append((R, C.flatten()))
    
    # Return refined camera poses and 3D points
    return refined_cameras, point_params

def run_bundle_adjustment(cameras, K, points_3d, feature_flags, x_features, y_features, visible_indices):
    
    n_cameras = len(cameras)
    n_points = len(points_3d)
    
    # Create lists for observations
    camera_indices = []
    point_indices = []
    points_2d = []
    
    # Collect all observations
    for i, point_idx in enumerate(visible_indices):
        for cam_idx in range(n_cameras):
            if feature_flags[point_idx, cam_idx] == 1:
                camera_indices.append(cam_idx)
                point_indices.append(i)
                points_2d.append([x_features[point_idx, cam_idx], y_features[point_idx, cam_idx]])
    
    # Convert to numpy arrays
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    points_2d = np.array(points_2d)
    
    print(f"Bundle adjustment with {len(points_2d)} observations")
    
    # Run bundle adjustment
    refined_cameras, refined_points = bundle_adjustment(
        cameras, points_3d, points_2d, camera_indices, point_indices, K
    )
    
    # Calculate reprojection error after bundle adjustment
    total_error = 0
    count = 0
    
    for i, point_idx in enumerate(visible_indices):
        point_3d = refined_points[i]
        
        for cam_idx in range(n_cameras):
            if feature_flags[point_idx, cam_idx] == 1:
                R, C = refined_cameras[cam_idx]
                t = -R @ C
                
                # Project point
                point_3d_reshaped = point_3d.reshape(1, 3)
                projected_pt, _ = cv2.projectPoints(point_3d_reshaped, 
                                                   cv2.Rodrigues(R)[0], 
                                                   t.reshape(3, 1), 
                                                   K, None)
                projected_pt = projected_pt.flatten()
                
                # Observed point
                observed_pt = np.array([x_features[point_idx, cam_idx], 
                                       y_features[point_idx, cam_idx]])
                
                # Calculate error
                error = np.linalg.norm(projected_pt - observed_pt)
                total_error += error
                count += 1
    
    if count > 0:
        mean_error = total_error / count
        print(f"Mean reprojection error after bundle adjustment: {mean_error:.3f} pixels")
    
    return refined_cameras, refined_points