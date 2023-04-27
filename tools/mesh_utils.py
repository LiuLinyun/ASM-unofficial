import torch

class MeshUVProjecter():
    def __init__(self, verts, tri_indices, uv_coords, uv_indices):
        self.tri_verts = verts[tri_indices] # (tri_cnt, 3, 3)
        self.tri_uvs = uv_coords[uv_indices] # (tri_cnt, 3, 2)
        self.tri_uv_center = torch.sum(self.tri_uvs, dim=1)

    def uv2mesh(self, uvs):
        # 找到 uv 坐标最近的三角形
        dist = torch.cdist(uvs, self.tri_uv_center) # (N,2) vs (T,2) -> (N,T)
        tri_idx = torch.argmin(dist, dim=1) # (N)
        select_uvs = self.tri_uvs[tri_idx] # (N, 3, 2)
        uv_cb = select_uvs[:,1] - select_uvs[:,2]
        uv_ca = select_uvs[:,0] - select_uvs[:,2]
        denominator = torch.cross(uv_ca, uv_cb, dim=-1)
        
        uv_cp = uvs - select_uvs[:,2]
        alpha = torch.cross(uv_cp, uv_cb, dim=-1) / denominator
        beta = torch.cross(uv_cp, uv_ca, dim=-1) / denominator
        gamma = 1 - (alpha + beta)
        bary = torch.cat([alpha.view(-1,1), beta.view(-1,1), gamma.view(-1,1)], dim=1).view(-1, 3, 1)

        tri = self.tri_verts[tri_idx] # (N, 3, 3)
        return torch.sum(bary*tri, dim=1)
