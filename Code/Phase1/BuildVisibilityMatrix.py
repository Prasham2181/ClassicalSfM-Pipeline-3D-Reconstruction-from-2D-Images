import numpy as np


"""The visibility matrix tells you which cameras see which 3D points, 
enabling you to compute reprojection errors only for observed points and to exploit the sparse nature of these observations.

"""

#  size is IxJ  i is number of camera and J is 3d points in the reconstruction


def build_visibility_matrix(feature_flags, valid_indices):
    sub_feature_flags = feature_flags[valid_indices, :]
    sub_feature_flags = (sub_feature_flags > 0).astype(int)   # only binary values aave 
    V = sub_feature_flags.T
    return V
