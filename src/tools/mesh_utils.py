import torch
from icecream import ic

def cross2d(x, y):
    # X (*,2), Y (*,2) -> (*)
    return x[...,0]*y[...,1] - x[...,1]*y[...,0]

class MeshUVProjecter():
    def __init__(self, verts, tri_indices, uv_coords, uv_indices):
        self.tri_verts = verts[tri_indices] # (tri_cnt, 3, 3)
        self.tri_uvs = uv_coords[uv_indices] # (tri_cnt, 3, 2)
        self.tri_uv_center = torch.sum(self.tri_uvs, dim=1)/3

    def uv2mesh(self, uvs):
        # 找到 uv 坐标最近的三角形
        dist = torch.cdist(uvs, self.tri_uv_center) # (N,2) vs (T,2) -> (N,T)
        tri_idx = torch.argmin(dist, dim=1) # (N)
        select_uvs = self.tri_uvs[tri_idx] # (N, 3, 2)
        a, b, c = select_uvs[:,0], select_uvs[:,1], select_uvs[:,2]
        ab, ac, ap = b-a, c-a, uvs-a
        s_total, s1, s2 = cross2d(ab, ac), cross2d(ap, ac), cross2d(ab, ap)
        alpha = s2 / s_total
        beta = s1 / s_total
        gamma = 1 - (alpha + beta)
        bary = torch.cat([alpha.view(-1,1), beta.view(-1,1), gamma.view(-1,1)], dim=1).view(-1, 3, 1)

        tri = self.tri_verts[tri_idx] # (N, 3, 3)
        return torch.sum(bary*tri, dim=1)

if __name__ == "__main__":

    import torch

    def cross2d(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    def barycentric_coordinates_cross_product(A, B, C, P):
        AB = B - A
        AC = C - A
        AP = P - A

        s_total = cross2d(AB, AC)
        s1 = cross2d(AP, AC)
        s2 = cross2d(AB, AP)

        alpha = s2 / s_total
        beta = s1 / s_total
        gamma = 1 - alpha - beta

        return alpha, beta, gamma

    # Example
    A = torch.tensor([0.5000, 0.5693])
    B = torch.tensor([0.5000, 0.5729])
    C = torch.tensor([0.4960, 0.5730])
    P = torch.tensor([0.5000, 0.5729])

    alpha, beta, gamma = barycentric_coordinates_cross_product(A, B, C, P)
    print(f"alpha: {alpha}, beta: {beta}, gamma: {gamma}")

