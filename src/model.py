import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch3d import transforms as trans

import sys
sys.path.append('.')
from tools.init_mu_face_rig import InitMuFaceRig
from tools.mesh_utils import MeshUVProjecter
from tools.face_dataset import basis_exp

from icecream import ic

def gen_face_with_expression(mu_face):
    E = basis_exp.shape[0]
    face = mu_face + basis_exp[random.randint(0,E)]
    return torch.tensor(face)
    
class VirtDataset(Dataset):
    def __init__(self):
        super().__init__()
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return idx

class AsmModel(pl.LightningModule):
    def __init__(self, K, init_mu_face_rig, base_verts, tgt_verts):
        super().__init__()
        J = len(init_mu_face_rig.bones_name)
        self.J = J
        self.K = K 
        self.B = tgt_verts.size() # batch_sie
        self.base_verts = base_verts
        self.tgt_verts = tgt_verts
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

    def gmm_weighting(self):
        # (V, 2) V means verts_cnt
        vert_uvs = self.init_state.uv_coords[self.init_state.verts_uv_indices]
        mu = (self.mu + self.zeta) # (J, K, 2)
        # x-mu: (V, 1, 2) - (J*K, 2) -> (V, J*K, 2)
        x_mu = vert_uvs.view(-1,1,2) - mu.view(-1,2)
        # (J, K, 1) -> (J, K, 4) -> (J, K, 2, 2)
        sigma_mat = torch.cat([self.sigma[:,:,[1]], self.sigma[:,:,[0]], 
            self.sigma[:,:,[0]], self.sigma[:,:,[2]]], dim = -1).view(self.J, self.K, 2, 2)
        # 1/(2*pi*|Sigma|^0.5) , shape (J, K)
        inv_2pi_sqrt_sigma_det = 1 / ((2*torch.pi) * torch.sqrt(torch.det(sigma_mat)))
        pi = self.pi / torch.sum(self.pi, dim=1, keepdim=True) # (J, K), normalize pi
        # (J,K) -> (J*K, 1) 
        pi_alpha = (pi*inv_2pi_sqrt_sigma_det).view(-1,1)
        # (V, J*K, 1, 2) @ (J*K,2,2) @ (V, J*K, 2, 1) -> (V, J*K, 1)
        index = x_mu.unsqueeze(-2).matmul(sigma_mat.view(-1,2,2)).matmul(x_mu.unsqueeze(-1)).squeeze(-1) / (-2)
        # (V, J, K)
        w = torch.mul(pi_alpha, torch.exp(index)).reshape(-1, self.J, self.K)
        wg = torch.div(w, torch.sum(w, dim=-1, keepdim=True)).squeeze(-1) # (V, J)
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
        # 但是实际计算只需要骨骼顶点 psi 的变化量 delta_psi
        # 更新的绑定姿态矩阵 B' = TB
        # T 为世界坐标系下骨骼的变换(这里只有平移)
        # B 就是骨骼原绑定姿态的local2world matrix
        vt = self.init_state.verts[self.init_state.bones_tail_idx]
        # (J, 3)
        delta_psi = self.mesh_uv_projecter.uv2mesh(self.zeta.view(-1,2)) - vt
        # (J,4,4)
        B = self.init_state.bones_M_local2obj
        # 添加平移 
        # B[:,:3,3] += delta_psi
        B_ = torch.empty_like(B)
        B_[:,:,:3] = B[:,:,:3]
        B_[:,:,3] = B[:,:,3]
        B_[:,:3,3] = B[:,:3,3] + delta_psi
        B_inv = B_.inverse()
        return B_inv

    def update_verts(self):
        W = self.gmm_weighting() # (V, J)
        M_l2w = self.bones_local2world() # (J, 4, 4)
        B_inv = self.dyn_binding() # (J, 4, 4)
        v = self.init_state.verts # (V,3)
        combined_trans = trans.Transform3d(matrix=torch.matmul(M_l2w,B_inv).transpose(-1,-2))
        v_weighted = combined_trans.transform_points(v)
        v_updated = torch.sum(v_weighted, dim=0)
        return v_updated
    
    def forward(self, x):
        v = self.update_verts()
        return v

    def training_step(self, batch, batch_idx):
        v = self(None)
        loss = F.mse_loss(v, self.tgt_verts)
        self.log("loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

if __name__ == "__main__":
    muface_rig = InitMuFaceRig("data/rig_info.json", "data/mu_face.obj")
    tgt_verts = gen_face_with_expression(muface_rig.verts)
    model = AsmModel(15, muface_rig, base_verts=muface_rig.verts, tgt_verts=tgt_verts)
    M1 = model.bones_local2world()
    M2 = model.dyn_binding()
    v = model.init_state.verts
    v_ = model.update_verts()
    cos_dist = torch.cosine_similarity(v, v_)
    ic(cos_dist)

