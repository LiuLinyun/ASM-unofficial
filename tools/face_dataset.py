import numpy as np
from scipy.io import loadmat

'''
mu_shape, shape: (1, 61443)
basis_shape, shape: (526, 61443)
EVs, shape: (526, 1)
basis_exp, shape: (203, 61443)
tri, shape: (40832, 3)
mask_face, shape: (1, 20481)
keypoints, shape: (1, 86)
tri_vt, shape: (40832, 3)
vt_list, shape: (20792, 2)
'''

matfile_path = "data/hifi3dpp.mat"
mat_content = loadmat(matfile_path)
# for k, v in mat_content.items():
#     print(f"{k}, shape: {(v.shape, str(v.dtype)) if isinstance(v, np.ndarray) else None}")

mu_shape = mat_content["mu_shape"].reshape(-1,3).astype(np.float32)
basis_shape = mat_content["basis_shape"].reshape(526,-1,3).astype(np.float32)
evs = mat_content["EVs"].reshape(-1).astype(np.float32) # (526,) ??what is it??
basis_exp = mat_content["basis_exp"].reshape(203,-1,3).astype(np.float32)
tri = mat_content["tri"].astype(np.int32)
mask_face = mat_content["mask_face"].reshape(-1).astype(bool)
keypoints = mat_content["keypoints"].reshape(-1).astype(np.int32) # (86,)
tri_vt = mat_content["tri_vt"].astype(np.int32)
vt_list = mat_content["vt_list"].astype(np.float32)

