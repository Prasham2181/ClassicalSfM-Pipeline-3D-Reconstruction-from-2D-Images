import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Create a folder for saving visualizations
output_dir = "SfM_Visualizations"
os.makedirs(output_dir, exist_ok=True)

def enforce_rank_2(F):
    """Enforces rank 2 on a fundamental matrix F."""
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Force rank 2 by setting the smallest singular value to zero
    return U @ np.diag(S) @ Vt

def save_figure(fig, filename):
    """ Save the figure to the output directory """
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight")
    print(f"Saved: {filepath}")

def compute_projection(P, X):
    """
    Project 3D points onto the image plane using a projection matrix.
    """
    if X.ndim == 1:
        X_h = np.hstack((X, 1))
    else:
        ones = np.ones((X.shape[0], 1))
        X_h = np.hstack((X, ones))

    x_proj_h = (P @ X_h.T).T  
    x_proj = x_proj_h[:, :2] / x_proj_h[:, 2:3]
    return x_proj

def visualize_F_matrix(F_rank3, F_rank2):
    """Visualize and compare F matrix before and after enforcing rank 2."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(F_rank3, cmap='coolwarm', interpolation='nearest')
    axs[0].set_title("F Matrix (Rank 3)")
    axs[1].imshow(F_rank2, cmap='coolwarm', interpolation='nearest')
    axs[1].set_title("F Matrix (Rank 2)")
    save_figure(fig, "F_Matrix_Comparison.png")
    plt.show()

def visualize_feature_matching(img1, img2, pts1, pts2, inliers):
    """ Visualize feature matching after RANSAC. """
    img_matches = np.hstack((img1, img2))
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        color = "g" if inliers[i] else "r"
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0] + img1.shape[1]), int(p2[1])
        ax.plot([x1, x2], [y1, y2], color, linewidth=1)
        ax.scatter([x1, x2], [y1, y2], color=color, s=10)

    ax.set_title("Feature Matching after RANSAC (Green: Inliers, Red: Outliers)")
    ax.axis("off")
    save_figure(fig, "Feature_Matching_RANSAC.png")
    plt.show()

def visualize_initial_triangulation(X_sets, C_sets):
    """Plot all four possible triangulated poses before disambiguation."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['r', 'g', 'b', 'y']
    for i, (X, C) in enumerate(zip(X_sets, C_sets)):
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors[i], label=f"Pose {i+1}")
        ax.scatter(C[0], C[1], C[2], c=colors[i], marker='x', s=100)

    ax.set_title("Initial Triangulation with Disambiguation")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    save_figure(fig, "Initial_Triangulation.png")
    plt.show()

def compare_triangulation(X_linear, X_nonlinear):
    """Compare 3D points from linear and nonlinear triangulation."""
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X_linear[:, 0], X_linear[:, 1], X_linear[:, 2], c='r', label='Linear')
    ax1.set_title("Linear Triangulation")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], X_nonlinear[:, 2], c='b', label='Nonlinear')
    ax2.set_title("Nonlinear Triangulation")

    plt.legend()
    save_figure(fig, "Triangulation_Comparison.png")
    plt.show()

def compare_projections(img1, img2, pts1_linear, pts2_linear, pts1_nonlinear, pts2_nonlinear):
    """Compare projections of linear vs nonlinear triangulation."""
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(img1, cmap='gray')
    axs[0, 0].scatter(pts1_linear[:, 0], pts1_linear[:, 1], c='r', s=10)
    axs[0, 0].set_title("Linear Triangulation - Image 1")

    axs[0, 1].imshow(img2, cmap='gray')
    axs[0, 1].scatter(pts2_linear[:, 0], pts2_linear[:, 1], c='r', s=10)
    axs[0, 1].set_title("Linear Triangulation - Image 2")

    axs[1, 0].imshow(img1, cmap='gray')
    axs[1, 0].scatter(pts1_nonlinear[:, 0], pts1_nonlinear[:, 1], c='g', s=10)
    axs[1, 0].set_title("Nonlinear Triangulation - Image 1")

    axs[1, 1].imshow(img2, cmap='gray')
    axs[1, 1].scatter(pts2_nonlinear[:, 0], pts2_nonlinear[:, 1], c='g', s=10)
    axs[1, 1].set_title("Nonlinear Triangulation - Image 2")

    save_figure(fig, "Projection_Comparison.png")
    plt.show()

def plot_triangulation_comparison(pts1, pts2, best_X, X_refined, K, R1, C1, best_R, best_C, img1= None, img2= None):
    """
    Visualizes the reprojected points using linear vs. non-linear triangulation.
    """
    P1 = K @ np.hstack((R1, -R1 @ C1.reshape(3,1)))  
    P2 = K @ np.hstack((best_R, -best_R @ best_C.reshape(3,1)))

    pts1_linear = compute_projection(P1, best_X)
    pts2_linear = compute_projection(P2, best_X)
    
    pts1_nonlinear = compute_projection(P1, X_refined)
    pts2_nonlinear = compute_projection(P2, X_refined)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    use_img_background = (img1 is not None and img2 is not None)

    # Linear triangulation - Image 1
    if use_img_background:
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

    axes[0, 0].scatter(pts1[:, 0], pts1[:, 1], c='g', label="Feature detections")
    axes[0, 0].scatter(pts1_linear[:, 0], pts1_linear[:, 1], c='r', marker='x', label="Reprojection (Linear)")
    axes[0, 0].set_title("Linear Triangulation (Image 1)")
    
    # Linear triangulation - Image 2
    if use_img_background:
        axes[0, 0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].scatter(pts2[:, 0], pts2[:, 1], c='g', label="Feature detections")
    axes[0, 1].scatter(pts2_linear[:, 0], pts2_linear[:, 1], c='r', marker='x', label="Reprojection (Linear)")
    axes[0, 1].set_title("Linear Triangulation (Image 2)")
    
    # Non-linear triangulation - Image 1
    if use_img_background:
        axes[1, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[1, 0].scatter(pts1[:, 0], pts1[:, 1], c='g', label="Feature detections")
    axes[1, 0].scatter(pts1_nonlinear[:, 0], pts1_nonlinear[:, 1], c='b', marker='x', label="Reprojection (Non-linear)")
    axes[1, 0].set_title("Non-Linear Triangulation (Image 1)")
    
    # Non-linear triangulation - Image 2
    if use_img_background:
        axes[1, 0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1, 1].scatter(pts2[:, 0], pts2[:, 1], c='g', label="Feature detections")
    axes[1, 1].scatter(pts2_nonlinear[:, 0], pts2_nonlinear[:, 1], c='b', marker='x', label="Reprojection (Non-linear)")
    axes[1, 1].set_title("Non-Linear Triangulation (Image 2)")

    for ax in axes.flat:
        ax.legend(loc='upper right', fontsize='x-small')
    
    plt.suptitle("Comparison: Linear vs. Non-Linear Triangulation")
    plt.tight_layout()
    plt.show()


def plot_pnp_ransac_comparison(pts_obs, pts_proj_initial, pts_proj_refined, img= None):
    """
    Plots the observed 2D points along with the reprojected points using
    the initial PnP (e.g., RANSAC) pose and the refined PnP pose.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    if img is not None:
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Initial PnP projection
    axes[0].scatter(pts_obs[:, 0], pts_obs[:, 1], c='g', marker='o', s=10, label="Observed")
    axes[0].scatter(pts_proj_initial[:, 0], pts_proj_initial[:, 1], c='r', marker='x', s=10, label="Initial PnP")
    axes[0].set_title("Initial PnP Projection")
    
    # Refined PnP projection
    axes[1].scatter(pts_obs[:, 0], pts_obs[:, 1], c='g', marker='o', s=10, label="Observed")
    axes[1].scatter(pts_proj_refined[:, 0], pts_proj_refined[:, 1], c='b', marker='x', s=10, label="Refined PnP")
    axes[1].set_title("Refined PnP Projection")
    
    plt.suptitle("PnP RANSAC: Observed vs. Reprojected Points")

    for ax in axes:
        ax.legend(loc='upper right')
    
    plt.tight_layout()

    plt.show()

def draw_epipolar_lines(img1, img2, pts1, pts2, F):
    """
    Draws epipolar lines on two images given a fundamental matrix F.
    """

    # Check if images are valid
    if img1 is None or img2 is None:
        raise ValueError("Cannot draw epipolar lines - images not available")

    img1 = img1.copy()
    img2 = img2.copy()
    
    # Convert keypoints to homogeneous coordinates
    pts1_h = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_h = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    
    # Compute epilines for pts1 (lines in image 2)
    lines2 = (F @ pts1_h.T).T
    # Compute epilines for pts2 (lines in image 1)
    lines1 = (F.T @ pts2_h.T).T

    # Function to draw epipolar lines
    def draw_lines(img, lines, pts):
        h, w, _ = img.shape
        img_with_lines = img.copy()
        for r, pt in zip(lines, pts):
            x0, y0 = map(int, [0, -r[2] / r[1]])  # Intersect with left edge
            x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])  # Intersect with right edge
            cv2.line(img_with_lines, (x0, y0), (x1, y1), (0, 0, 0), 1)
            cv2.circle(img_with_lines, tuple(pt.astype(int)), 5, (255, 255, 255), -1)
        return img_with_lines

    img1_with_lines = draw_lines(img1, lines1, pts1)
    img2_with_lines = draw_lines(img2, lines2, pts2)

    return img1_with_lines, img2_with_lines

def visualize_F_matrix_comparison(img1, img2, pts1, pts2, F):
    """
    Plots the fundamental matrix comparison between rank 3 and rank 2.
    """
    # Check if images are valid
    if img1 is None or img2 is None:
        print("Cannot visualize F matrix comparison - images not available")
        return
        
    F_rank2 = enforce_rank_2(F)  # Enforce rank 2 on F

    # Compute epipolar lines for both cases
    img1_F3, img2_F3 = draw_epipolar_lines(img1, img2, pts1, pts2, F)
    img1_F2, img2_F2 = draw_epipolar_lines(img1, img2, pts1, pts2, F_rank2)

    # Plot images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(cv2.cvtColor(img2_F3, cv2.COLOR_BGR2RGB))
    axes[0].set_title(r"$\mathbf{rank(F) = 3}$")

    axes[1].imshow(cv2.cvtColor(img2_F2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(r"$\mathbf{rank(F) = 2}$")

    plt.suptitle("F Matrix: Rank 3 vs Rank 2 Comparison")
    plt.show()
