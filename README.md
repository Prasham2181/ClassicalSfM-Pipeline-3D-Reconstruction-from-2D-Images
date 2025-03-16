# Structure from Motion (SfM) Implementation

This repository contains a comprehensive implementation of Structure from Motion (SfM) pipeline for 3D reconstruction from multiple images. The project implements the complete SfM pipeline from feature extraction to bundle adjustment.

## Authors

- Prasham Soni
- Sarthak Mehta

## Project Overview

Structure from Motion is a photogrammetric range imaging technique for estimating three-dimensional structures from two-dimensional image sequences that may be coupled with local motion signals. This implementation follows a traditional SfM pipeline, including:

1. Feature extraction and matching
2. Fundamental matrix estimation using RANSAC
3. Essential matrix computation
4. Camera pose extraction
5. Triangulation (linear and non-linear)
6. Perspective-n-Point (PnP) for additional view registration
7. Bundle adjustment for global refinement

## Project Structure

```

Group_19.zip 
│   README.md
|   Code 
|   ├── Phase1/
|   | ├── GetInliersRANSAC.py          # RANSAC for feature matching inliers
|   | ├── EstimateFundamentalMatrix.py # Fundamental matrix computation
|   | ├── EssentialMatrixFromFundamentalMatrix.py # E matrix from F matrix
|   | ├── ExtractCameraPose.py         # Extract camera poses from E matrix
|   | ├── LinearTriangulation.py       # Initial 3D point triangulation
|   | ├── DisambiguateCameraPose.py    # Camera pose disambiguation
|   | ├── NonlinearTriangulation.py    # Refined 3D point triangulation
|   | ├── PnPRANSAC.py                 # PnP with RANSAC for new views
|   | ├── NonlinearPnP.py              # Non-linear refinement of PnP
|   | ├── BuildVisibilityMatrix.py     # Build visibility matrix for BA
|   | ├── BundleAdjustment.py          # Bundle adjustment implementation
|   | ├── Wrapper.py                   # Main script that runs the pipeline
|   | Data
|   | └── IntermediateOutputImages/    # Visualization outputs

```

## Dependencies

- OpenCV (`cv2`)
- NumPy
- Matplotlib
- tqdm (for progress bars)

## Usage

To run the SfM pipeline:

```bash
cd Code/Phase1
python Wrapper.py
```

Make sure the data path in the `Wrapper.py` file is correctly set to point to your dataset folder.

## Pipeline Details

### 1. Feature Extraction

The implementation extracts and matches features from multiple images using the provided matching files. Feature correspondences across images are compiled into matrices for easy access.

### 2. Initial Reconstruction (Images 1 & 2)

- Compute the fundamental matrix (F) using RANSAC to handle outliers
- Compute the essential matrix (E) using the camera intrinsic matrix (K)
- Extract possible camera poses from E
- Disambiguate the correct camera pose using triangulation and chirality constraint

### 3. 3D Point Triangulation

- Linear triangulation to obtain initial 3D point estimates
- Non-linear refinement to minimize reprojection error
- Filter points based on reprojection error threshold

### 4. Adding New Images via PnP

- For each new image, compute the camera pose using PnP RANSAC
- Refine the pose using non-linear optimization
- Triangulate new points visible in the registered image

### 5. Bundle Adjustment

- Global optimization of camera poses and 3D points
- Minimization of reprojection error across all cameras and points
- Visualization of pre- and post-BA reconstruction

## Visualizations

The code includes several visualization functions:

- Top view of reconstructed 3D points and camera centers
- Chirality check to verify points are in front of cameras
- Comparison between pre-BA and post-BA reconstruction
- Reprojection error visualizations

## Performance Metrics

The implementation reports various metrics for evaluation:

- Mean reprojection error for linear triangulation
- Mean reprojection error for non-linear triangulation
- Mean reprojection error for PnP (linear and non-linear)
- Chirality statistics (number of points in front of cameras)

## Notes

- The camera intrinsic matrix K is provided in the code
- The implementation handles rank-2 enforcement for the fundamental matrix
- A 24-hour SLA is maintained for agent responses
- The code includes comprehensive error checking and reporting
