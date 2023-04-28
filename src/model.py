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
        self.zeta = nn.Parameter(torch.zeros(J,1,2), requires_grad=True)
        self.pi = nn.Parameter(torch.ones(J,K), requires_grad=True)
        self.mu = nn.Parameter(torch.zeros(J,K,2), requires_grad=True)
        # 三个数分别表示正态分布协方差矩阵中的三个数 [rho*sigma_x*sigma_y, sigma_x^2, sigma_y^2] 
        self.sigma = nn.Parameter(torch.ones(J,K,3), requires_grad=True)
        self.tau_scale = nn.Parameter(torch.ones(J,3), requires_grad=True)
        # 用 so3 表示旋转
        self.tau_so3 = nn.Parameter(torch.zeros(J,3), requires_grad=True)
        self.tau_translation = nn.Parameter(torch.zeros(J,3), requires_grad=True)

        # 初始化参数并预处理一些数据
        self.init_params()
        self.pre_processing()

    def init_params(self):
        pass

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
        # (J,K) -> (J*K, 1) TODO 归一化 pi
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
        recursive_trans('face', trans.Transform3d(), trans_mat_dict)
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
    model.bones_local2world()
    model.dyn_binding()
