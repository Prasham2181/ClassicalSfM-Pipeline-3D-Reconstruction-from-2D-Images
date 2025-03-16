# Create Your Own Starter Code :)

# Prasham Soni
# Sarthak Mehta
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import tqdm

from GetInliersRANSAC import *
from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import * 
from NonlinearTriangulation import * 
from LinearPnP import *
from PnPRANSAC import *
from NonlinearPnP import *
from BuildVisibilityMatrix import *
from BundleAdjustment import *

from plot_triangulations import *

def enforce_rank_2(F):
    """
    Ensures that the Fundamental Matrix F has rank 2 by setting the smallest singular value to zero.
    """
    U, S, Vt = np.linalg.svd(F)
    print("Original singular values:", S)
    print("Original matrix rank:", np.linalg.matrix_rank(F))
    
    S[-1] = 0  # Force the last singular value to zero
    F_rank2 = U @ np.diag(S) @ Vt
    
    print("New matrix rank:", np.linalg.matrix_rank(F_rank2))
    print("Difference norm:", np.linalg.norm(F - F_rank2))
    
    return F_rank2

def extract_features(data):
    num_images = 5  
    rgb_features = []   
    x_features = []     
    y_features = []     
    feature_flags = []  

    # There are matching files for the first 4 images
    for n in range(1, num_images):  
        filename = os.path.join(data, "matching" + str(n) + ".txt")
        with open(filename, "r") as f:
            header = f.readline().strip()
            num_features = int(header.split(':')[1].strip())
            
            for line in f:
                tokens = line.split()
                if len(tokens) == 0:
                    continue  
                
                n_matches = int(tokens[0])
                
                r_val = float(tokens[1])
                g_val = float(tokens[2])
                b_val = float(tokens[3])
                rgb_features.append([r_val, g_val, b_val])
                
                current_x = float(tokens[4])
                current_y = float(tokens[5])
                
                x_row = np.zeros(num_images)
                y_row = np.zeros(num_images)
                flag_row = np.zeros(num_images, dtype=int)
                
                # Store current image's feature location.
                x_row[n-1] = current_x
                y_row[n-1] = current_y
                flag_row[n-1] = 1
                
                m = 1  # token offset for additional matches
                while n_matches > 1:
                   
                    image_id = int(tokens[5 + m])
                    match_x = float(tokens[6 + m])
                    match_y = float(tokens[7 + m])
                    x_row[image_id - 1] = match_x  
                    y_row[image_id - 1] = match_y
                    flag_row[image_id - 1] = 1
                    m += 3
                    n_matches -= 1
                
                x_features.append(x_row)
                y_features.append(y_row)
                feature_flags.append(flag_row)
    
    x_features = np.array(x_features)
    y_features = np.array(y_features)
    feature_flags = np.array(feature_flags)
    rgb_features = np.array(rgb_features)
    
    return x_features, y_features, feature_flags, rgb_features

def compute_mean_reprojection_error(X_points, P1, P2, pts1, pts2):

    if isinstance(X_points, tuple):
        X_points, inlier_mask = X_points
        # Only use inlier points if a mask is provided
        if inlier_mask is not None:
            X_points = X_points[inlier_mask]
            pts1 = pts1[inlier_mask]
            pts2 = pts2[inlier_mask]
    else:
        X_points = X_points

    total_error = 0.0
    N = X_points.shape[0]
    for i in range(N):
        # Compute error for camera 1:
        res1 = reprojection_error(X_points[i], P1, pts1[i])
        err1 = np.linalg.norm(res1)
        # Compute error for camera 2:
        res2 = reprojection_error(X_points[i], P2, pts2[i])
        err2 = np.linalg.norm(res2)
        total_error += (err1 + err2)
    mean_error = total_error / N
    return mean_error

# ------------------------ New Visualization Functions ------------------------

def visualize_matches(image, pts_obs, pts_proj, title="Matches"):
    """
    Displays the image with observed 2D points and projected 3D points.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='gray')
    plt.scatter(pts_obs[:, 0], pts_obs[:, 1], s=40, c='red', marker='o', label="Observed")
    plt.scatter(pts_proj[:, 0], pts_proj[:, 1], s=40, c='green', marker='x', label="Projected")
    plt.legend()
    plt.title(title)
    plt.show()

def plot_top_view(cameras, points_3d, title="Top View of 3D Reconstruction"):
    """
    Plots a top view (x vs. z) of the 3D point cloud and the camera centers.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot 3D points (using x and z coordinates)
    plt.scatter(points_3d[:, 0], points_3d[:, 2], s=1, c='blue', label='3D Points')
    
    # Plot each camera center with a red triangle
    for idx, cam in enumerate(cameras):
        plt.scatter(cam[0], cam[2], c='red', marker='^', s=100, label="Camera Center" if idx == 0 else "")
        plt.text(cam[0], cam[2], f"{idx+1}", fontsize=12, color='black')
    
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_chirality(camera_center, R, points_3d, title="Chirality Check (Top View)"):
    """
    Plots a top-view (X vs. Z) of 3D points, colored by whether they are in front of or behind the camera.
    """
    # Transform points to camera coordinates: X_cam = R*(X - C)
    depths = np.array([(R @ (point - camera_center))[2] for point in points_3d])
    
    # Separate points: in front if depth > 0, behind otherwise.
    front_points = points_3d[depths > 0]
    behind_points = points_3d[depths <= 0]
    
    plt.figure(figsize=(10,8))
    plt.scatter(front_points[:, 0], front_points[:, 2], s=1, c='green', label='In Front (Z > 0)')
    plt.scatter(behind_points[:, 0], behind_points[:, 2], s=1, c='red', label='Behind (Z <= 0)')
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def check_chirality(camera_center, R, points_3d):
   
    depths = np.array([(R @ (point - camera_center))[2] for point in points_3d])
    num_in_front = np.sum(depths > 0)
    num_total = len(depths)
    return num_in_front, num_total, depths 

# --------------------- PnP Registration Function ---------------------

def register_new_image_pnp(new_idx, x_features, y_features, feature_flags, K, existing_3D_points, valid_indices):
    sub_feature_flags = feature_flags[valid_indices, :]
    valid_new = (sub_feature_flags[:, new_idx] == 1)
    
    pts_new = np.stack([
        x_features[valid_indices[valid_new], new_idx],
        y_features[valid_indices[valid_new], new_idx]
    ], axis=1)
    
    pts_3D = existing_3D_points[valid_new]
    
    # Call the updated PnP RANSAC routine.
    # It returns (R, C, inliers) directly. No extra conversion is needed.
    new_R, new_C, inliers = get_pnp_Ransac(pts_3D, pts_new, K, num_iterations=5000, reproj_threshold=2.0)
    
    return new_C, new_R, pts_new, valid_new



def plot_top_view_comparison(cameras, points_pre, points_post, title="Top View: Pre-BA vs. Post-BA"):

    plt.figure(figsize=(10, 8))
    
    # Plot pre–BA points in blue.
    plt.scatter(points_pre[:, 0], points_pre[:, 2], s=1, c='blue', label='Pre-BA Points')
    
    # Plot post–BA points in green.
    plt.scatter(points_post[:, 0], points_post[:, 2], s=1, c='green', label='Post-BA Points')
    
    # Plot camera centers in red.
    for idx, cam in enumerate(cameras):
        plt.scatter(cam[0], cam[2], c='red', marker='^', s=100, label="Camera Center" if idx==0 else "")
        plt.text(cam[0], cam[2], f"{idx+1}", fontsize=12, color='black')
    
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()




def main():
    # data_path = r"C:\Users\sonip\Desktop\WPI\Sem2_spring_2025\Computer vision\Homeworks\P2\YourDirectoryID_p2\YourDirectoryID_p2\Phase1\Phase1\P2Data\P2Data"
    data_path = r"Code\Phase1\P2Data\P2Data"
    # data_path = "/home/sarthak_m/ComputerVision/P2_SfM_and_NeRf/Phase1/P2Data/P2Data"
    ENABLE_VISUALIZATIONS = True

    # ------------------- Feature Extraction -------------------
    x_features, y_features, feature_flags, rgb_features = extract_features(data_path)
    print("******************** FEATURE EXTRACTION COMPLETE **********************")
    
    # Use images 1 and 2 for initial reconstruction.
    valid = (feature_flags[:, 0] == 1) & (feature_flags[:, 1] == 1)
    pts1 = np.stack([x_features[valid, 0], y_features[valid, 0]], axis=1)
    pts2 = np.stack([x_features[valid, 1], y_features[valid, 1]], axis=1)
    print(f"Points selected are {pts1}, and {pts2}")
    print("Number of correspondences between images 1 and 2:", pts1.shape[0])
    
    valid_indices = np.where(valid)[0]
    pts1_cv = pts1.astype(np.float32)
    pts2_cv = pts2.astype(np.float32)
    num_iterations = 5000

    F, inlier_mask = cv2.findFundamentalMat(
        pts1_cv, pts2_cv, 
        method=cv2.FM_RANSAC, 
        ransacReprojThreshold=1.0, 
        confidence=0.99,
        maxIters=num_iterations
    )
    
    if inlier_mask is not None:
        inlier_mask = inlier_mask.ravel().astype(bool)
        inlier_count = np.sum(inlier_mask)
    else:
        print("Warning: findFundamentalMat failed, using fallback method")
        F, inlier_count = getInliners(pts1, pts2, num_iterations, threshold)
    
    print(f"Number of inliers: {inlier_count} out of {pts1.shape[0]}")
    print("**********************CHECKING THE RANK 2 ASPECT OF F MATRIX **********")
    U, S, Vt = np.linalg.svd(F)
    print("Singular values:", S)
    print("Estimated Fundamental Matrix:\n", F)
    # visualize_F_matrix(F, enforce_rank_2(F))
    print("Number of inliers:", inlier_count)
    print("******************** FUNDAMENTAL MATRIX COMPUTED **********************")
    
    # ------------------- Essential Matrix & Initial Reconstruction -------------------
    K = np.array([[531.122155322710, 0, 407.192550839899],
                  [0, 531.541737503901, 313.308715048366],
                  [0, 0, 1]])
    E = compute_essential_matrix(F, K)
    print(f"Essential matrix E : {E}")
    print("*********************** ESSENTIAL MATRIX COMPUTED **********************")
    
    C1 = np.array([0, 0, 0])
    R1 = np.eye(3)
    
    # Storage for all camera poses
    cameras_R = {0: R1}
    cameras_C = {0: C1}
    
    camera_pose = extract_camera_pose(E)
    C_set, R_set, X_set = [], [], []
    for i, (C, R) in enumerate(camera_pose):
        print(f"*********************** CAMERA POSE {i} ESTIMATED **********************")
        print(f"Configuration {i+1}:")
        print("Camera center (up to scale):", C)
        print("Rotation matrix:\n", R)
        print("Determinant of R:", np.linalg.det(R))
        print()
        X, inlier_mask = linearTriangulation(K, C1, R1, C, R, pts1, pts2)
        print("Triangulated 3D points:\n", X)
        print("\n")
        C_set.append(C)
        R_set.append(R)
        X_set.append(X)
        print("Triangulated points shape:", X.shape)
        print("Number of points with positive depth:", np.sum(inlier_mask))
    
    print("*********************** All CAMERA POSEs ESTIMATED & LINEAR TRIANGULATION DONE **********************")
    best_C, best_R, best_X = disambiguateCameraPose(np.array(C_set), np.array(R_set), X_set)
    print("\n********** DISAMBIGUATED CAMERA POSE **********")
    print("Best Camera Center:", best_C)
    print("Best Rotation Matrix:\n", best_R)
    print("Triangulated 3D points from best candidate (first 5 points):\n", best_X[:5])
    print("\n********** DISAMBIGUATED CAMERA POSE DONE **********")
    
    P1 = projection_matrix_p(K, R1, C1)
    P2 = projection_matrix_p(K, best_R, best_C)
    
    good_point_indices = filter_points_by_error(best_X, P1, P2, pts1, pts2, threshold=3.0)
    print(f"Keeping {len(good_point_indices)} points with reprojection error < 3.0 pixels out of {best_X.shape[0]}")
    
    # After disambiguation
    cameras_R[1] = best_R
    cameras_C[1] = best_C
    good_X = best_X[good_point_indices]
    good_pts1 = pts1[good_point_indices]
    good_pts2 = pts2[good_point_indices]
    good_valid_indices = valid_indices[good_point_indices]
    
    mean_err_linear = compute_mean_reprojection_error(good_X, P1, P2, good_pts1, good_pts2)
    print(f"Mean reprojection error (Linear Triangulation): {mean_err_linear}")
    
    X_refined_all = np.zeros_like(best_X)
    X_refined_all[good_point_indices] = np.array([
        non_linearTriangulation(K, R1, C1, best_R, best_C, good_X[i], good_pts1[i], good_pts2[i])
        for i in range(len(good_point_indices))
    ])
    mean_err_nonlinear = compute_mean_reprojection_error(X_refined_all[good_point_indices], P1, P2, good_pts1, good_pts2)
    print(f"Mean reprojection error (Non-Linear Triangulation): {mean_err_nonlinear}")
    
    # --------------------- Visualization: Before Bundle Adjustment ---------------------
    pre_BA_camera_centers = np.array([cameras_C[idx] for idx in sorted(cameras_C.keys())])
    pre_BA_points = good_X.copy()  # Save pre–BA points for comparison later.
    
    plot_top_view(pre_BA_camera_centers, good_X, title="Top View Before Bundle Adjustment")
    num_in_front, total, depths = check_chirality(best_C, best_R, good_X)
    print(f"Chirality Check (Best Candidate, Before BA): {num_in_front} out of {total} points are in front of the camera.")
    plot_chirality(best_C, best_R, good_X, title="Chirality Check (Before BA)")
    
    # --------------------- New-Image Registration & PnP ---------------------
    linear_pnp_errors = []
    nonlinear_pnp_errors = []
    
    for new_image_index in range(2, 5):  # For images 3 to 5
        new_C, new_R, pts_new, valid_flags = register_new_image_pnp(new_image_index, 
                                                                     x_features, 
                                                                     y_features, 
                                                                     feature_flags, 
                                                                     K, 
                                                                     best_X, 
                                                                     valid_indices)
        print(f"\n********** REGISTERED NEW IMAGE (Image {new_image_index+1}) via PnP ************")
        print("Estimated Camera Center (Initial):", new_C)
        print("Estimated Rotation Matrix (Initial):\n", new_R)
        print("Number of 2D correspondences in new image:", pts_new.shape[0])
        
        sub_feature_flags = feature_flags[valid_indices, :]
        valid_new = (sub_feature_flags[:, new_image_index] == 1)
        pts_3D_for_refine = best_X[valid_new]
        pts_2D_for_refine = pts_new
        
        if pts_new.shape[0] < 6:
            print(f"Warning: Image {new_image_index+1} has too few correspondences ({pts_new.shape[0]}). Skipping refinement.")
            continue
        
        # Compute initial reprojection error for PnP (linear)
        t_initial = -new_R @ new_C  
        pts_proj_initial = project_points(pts_3D_for_refine, new_R, t_initial, K)
        error_initial = np.linalg.norm(pts_proj_initial - pts_2D_for_refine, axis=1).mean()
        print(f"Reprojection Error (Initial PnP) for Image {new_image_index+1}: {error_initial:.2f}")
        linear_pnp_errors.append(error_initial)
        
        # Refine the pose using nonlinear PnP
        refined_C, refined_R = nonlinearPnP(pts_3D_for_refine, pts_2D_for_refine, K, new_C, new_R)
        print(f"\n********** REFINE NEW IMAGE (Image {new_image_index+1}) via Nonlinear PnP ************")
        print("Refined Camera Center:", refined_C)
        print("Refined Rotation Matrix:\n", refined_R)
        
        t_refined = -refined_R @ refined_C
        pts_proj_refined = project_points(pts_3D_for_refine, refined_R, t_refined, K)
        error_refined = np.linalg.norm(pts_proj_refined - pts_2D_for_refine, axis=1).mean()
        print(f"Reprojection Error (Nonlinear PnP) for Image {new_image_index+1}: {error_refined:.2f}")
        nonlinear_pnp_errors.append(error_refined)
        
        pts2_for_triangulation = np.stack([
            x_features[valid_indices[valid_new], 1],
            y_features[valid_indices[valid_new], 1]
        ], axis=1)
        
        if pts2_for_triangulation.shape[0] != pts_new.shape[0]:
            print(f"Warning: Mismatch in triangulation correspondences for Image {new_image_index+1}. Skipping re-triangulation.")
            continue
        
        P2 = projection_matrix_p(K, best_R, best_C)  # Projection for 2nd image
        P_new_refined = projection_matrix_p(K, refined_R, refined_C)
        X_new = linearTriangulation(K, best_C, best_R, refined_C, refined_R, pts2_for_triangulation, pts_new)
        error_triangulated = compute_mean_reprojection_error(X_new, P2, P_new_refined, pts2_for_triangulation, pts_new)
        print(f"Reprojection Error (Triangulation using refined pose) for Image {new_image_index+1}: {error_triangulated:.2f}")
        
        # Store refined camera pose
        cameras_R[new_image_index] = refined_R
        cameras_C[new_image_index] = refined_C
        
        # Bundle Adjustment: collect all registered cameras so far
        all_cameras = [(R1, C1), (best_R, best_C)]
        for cam_idx in range(2, new_image_index+1):
            all_cameras.append((cameras_R[cam_idx], cameras_C[cam_idx]))
            
        visible_points = []
        visible_indices = []
        for i, idx in enumerate(good_valid_indices):
            visible_count = np.sum(feature_flags[idx, :new_image_index+1])
            if visible_count >= 2:
                visible_points.append(good_X[i])
                visible_indices.append(idx)
        visible_points = np.array(visible_points)
        
        print(f"\n********** RUNNING BUNDLE ADJUSTMENT with {len(all_cameras)} cameras ************")
        refined_cameras, refined_points = run_bundle_adjustment(
            all_cameras, K, visible_points, feature_flags, x_features, y_features, visible_indices
        )
        
        # Update camera poses with BA results
        R1, C1 = refined_cameras[0]
        best_R, best_C = refined_cameras[1]
        for cam_idx in range(2, new_image_index+1):
            cameras_R[cam_idx], cameras_C[cam_idx] = refined_cameras[cam_idx-2]
        for i, idx in enumerate(visible_indices):
            point_idx = np.where(good_valid_indices == idx)[0]
            if len(point_idx) > 0:
                good_X[point_idx[0]] = refined_points[i]
        
        print("Bundle adjustment complete. Cameras and 3D points refined.")
    
    # --------------------- Visualization: After Bundle Adjustment ---------------------
    final_camera_centers = np.array([cameras_C[idx] for idx in sorted(cameras_C.keys())])
    plot_top_view(final_camera_centers, good_X, title="Final Reconstruction (After BA)")
    num_in_front, total, depths = check_chirality(best_C, best_R, good_X)
    print(f"Chirality Check (After BA): {num_in_front} out of {total} points are in front of the camera.")
    plot_chirality(best_C, best_R, good_X, title="Chirality Check (After BA)")
    
    # --------------------- Combined Pre-BA vs. Post-BA Visualization ---------------------
    plot_top_view_comparison(final_camera_centers, pre_BA_points, good_X, title="Comparison: Pre-BA vs. Post-BA Reconstruction")
    
    # --------------------- Print Overall Mean Errors ---------------------
    overall_linear_pnp_error = np.mean(linear_pnp_errors) if linear_pnp_errors else None
    overall_nonlinear_pnp_error = np.mean(nonlinear_pnp_errors) if nonlinear_pnp_errors else None
    print("\n*****************************************************************************************************")
    print(f"Overall Mean Reprojection Error (Linear Triangulation): {mean_err_linear:.2f}")
    print(f"Overall Mean Reprojection Error (Non-Linear Triangulation): {mean_err_nonlinear:.2f}")
    if overall_linear_pnp_error is not None:
        print(f"Overall Mean Reprojection Error (Linear PnP): {overall_linear_pnp_error:.2f}")
    if overall_nonlinear_pnp_error is not None:
        print(f"Overall Mean Reprojection Error (Nonlinear PnP): {overall_nonlinear_pnp_error:.2f}")
    print("\n*****************************************************************************************************")
    
if __name__ == "__main__":
    main()



