import torch 
import torch.nn as nn
import torch.functional as F
import pytorch_lightning as pl
from pytorch3d import transforms as trans

import sys
sys.path.append('.')
from tools.init_mu_face_rig import InitMuFaceRig
from tools.mesh_utils import MeshUVProjecter

from icecream import ic

class AsmModel(pl.LightningModule):
    def __init__(self, K, init_mu_face_rig):
        super().__init__()
        J = len(init_mu_face_rig.bones_name)
        self.J = J
        self.K = K 
        self.init_state = init_mu_face_rig
        self.mesh_uv_projecter = MeshUVProjecter(
            self.init_state.verts,
            self.init_state.tri_indices,
            self.init_state.uv_coords,
            self.init_state.uv_indices
        )
        # 可学习的参数
        zeta, scale, so3, translation = self.init_params()
        self.zeta = nn.Parameter(zeta.view(J,1,2), requires_grad=True)
        self.pi = nn.Parameter(torch.ones(J,K), requires_grad=True)
        self.mu = nn.Parameter(torch.zeros(J,K,2), requires_grad=True)
        # 三个数分别表示正态分布协方差矩阵中的三个数 [rho*sigma_x*sigma_y, sigma_x^2, sigma_y^2] 
        self.sigma = nn.Parameter(torch.ones(J,K,3), requires_grad=True)
        self.tau_scale = nn.Parameter(scale, requires_grad=True)
        # 用 so3 表示旋转
        self.tau_so3 = nn.Parameter(so3, requires_grad=True)
        self.tau_translation = nn.Parameter(translation, requires_grad=True)

        # 预处理一些数据, 避免重复计算
        self.pre_processing()

    def init_params(self):
        # 获得骨骼初始的 UV 坐标, 以初始化 zeta
        zeta = self.init_state.uv_coords[self.init_state.verts_uv_indices[self.init_state.bones_tail_idx]]

        # 获得骨骼初始的局部变换矩阵 
        # B_{curr}^{l2w} = B_{parent}^{l2w} T_{curr}^{local}
        # 获得 T_{curr}^{local} 之后初始化 tau
        M_local_trans_list = []
        for name in self.init_state.bones_name:
            bone_info = self.init_state.bones_dict[name]
            parent_name = bone_info.parent
            parent_bone_info = self.init_state.bones_dict[parent_name]
            M_curr2world = bone_info.M_local2obj
            M_parent2world = parent_bone_info.M_local2obj
            T = torch.matmul(M_parent2world.inverse(), M_curr2world).unsqueeze(0)
            M_local_trans_list.append(T)
        M_local = torch.cat(M_local_trans_list, dim=0)
        # 分解变换矩阵, 得到位移, 旋转, 缩放
        sR = M_local[:,:3,:3]
        s = torch.norm(sR, dim=1, keepdim=True)
        R = sR / s
        so3 = trans.so3_log_map(R.transpose(-1,-2))
        T = M_local[:,:3,3]
        return zeta, s.view(-1,3), so3, T

    def pre_processing(self):
        self.psi_init = init_state.bones_tail_pos
        # (J, 3)
        self.vt = self.init_state.verts[self.init_state.bones_tail_idx]
        # (J, K, 1) -> (J, K, 4) -> (J, K, 2, 2)
        self.sigma_mat = torch.cat([self.sigma[:,:,[1]], self.sigma[:,:,[0]], 
            self.sigma[:,:,[0]], self.sigma[:,:,[2]]], dim = -1).view(self.J, self.K, 2, 2)
        # 1/(2*pi*|Sigma|^0.5) , shape (J, K)
        self.inv_2pi_sqrt_sigma_det = 1 / ((2*torch.pi) * torch.sqrt(torch.det(self.sigma_mat)))
        # (tri_cnt, 3, 2)
        # self.tri_uvs = self.init_state.uv_coords[self.init_state.uv_indices]
        # (V, 2) V means verts_cnt
        self.vert_uvs = self.init_state.uv_coords[self.init_state.verts_uv_indices]

    def gmm_weighting(self):
        mu = (self.mu + self.zeta) # (J, K, 2)
        # (V, 1, 2) - (J*K, 2) -> (V, J*K, 2)
        x_mu = self.vert_uvs.view(-1,1,2) - mu.view(-1,2)
        pi = torch.normalize(self.pi, dim=1) # (J, K)
        # (J,K) -> (J*K, 1) 
        pi_alpha = (pi*self.inv_2pi_sqrt_sigma_det).view(-1,1)
        # (V, J*K, 1, 2) @ (J*K,2,2) @ (V, J*K, 2, 1) -> (V, J*K, 1)
        index = x_mu.squeeze(-2).bmm(self.sigma_mat.view(-1,2,2)).bmm(x_mu.squeeze(-1)) / (-2)
        # (V, J, K)
        w = pi_alpha*torch.exp(index).reshape(-1, self.J, self.K)
        w = torch.sum(w, dim=-1) # (V, J)
        wg = w / torch.sum(w, dim=-1).squeeze(-1) # (V, J)
        return wg

    def bones_local2world(self):
        trans_mat_dict = {}
        def recursive_trans(name, trans_parent, trans_dict):
            if name != "face":
                bone_idx = self.init_state.bones_name_idx[name]
                trans_tau = trans.Scale(self.tau_scale[[bone_idx]]) \
                    .compose(trans.Rotate(trans.so3_exp_map(self.tau_so3[[bone_idx]]))) \
                    .compose(trans.Translate(self.tau_translation[[bone_idx]]))
                trans_current = trans_tau.compose(trans_parent)
                trans_dict[name] = trans_current.get_matrix().transpose(-1,-2)
            else:
                trans_current = trans_parent
            for child in self.init_state.bones_dict[name].children:
                recursive_trans(child, trans_current, trans_dict)
        M_root_local2obj = self.init_state.bones_dict["face"].M_local2obj
        recursive_trans('face', trans.Transform3d(matrix=M_root_local2obj.T), trans_mat_dict)
        # (J, 4, 4)
        M_local2world = torch.cat([trans_mat_dict[k] for k in self.init_state.bones_name], dim=0)
        return M_local2world
        
    
    def dyn_binding(self):
        # psi = self.mesh_uv_projecter.uv2mesh(self.zeta) - self.vt + self.psi_init
        # TODO: 更新 BindPose 变换矩阵 ? 怎么实现?
        # ANS: B' = TB, T 为世界坐标系下骨骼的变换, 这里只有平移
        # B 就是骨骼的 matrix
        # (J, 3)
        delta_psi = self.mesh_uv_projecter.uv2mesh(self.zeta.view(-1,2)) - self.vt
        # (J,4,4)
        B = self.init_state.bones_M_local2obj
        # 添加平移 
        B[:,:3,3] += delta_psi
        B_inv = B.inverse()
        return B_inv

if __name__ == "__main__":
    init_state = InitMuFaceRig("data/rig_info.json", "data/mu_face.obj")
    model = AsmModel(2, init_state)
    model.pre_processing()
    M1 = model.bones_local2world()
    M2 = model.dyn_binding()
    # ic(torch.bmm(M1,M2))
