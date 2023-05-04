import os
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch3d import transforms as trans

from tools.init_mu_face_rig import InitMuFaceRig
from tools.mesh_utils import MeshUVProjecter
from tools.face_dataset import basis_exp
from tools.read_write_obj import write_obj_file

from icecream import ic

class EmphasizedLoss(nn.Module):
    def __init__(self, power=2):
        super(EmphasizedLoss, self).__init__()
        self.power = power

    def forward(self, predictions, targets):
        # Calculate the squared differences between predictions and targets
        squared_diffs = (predictions - targets) ** 2

        # Apply the power factor to emphasize larger distances
        emphasized_loss = squared_diffs ** (self.power / 2)

        # Average the emphasized loss across the batch
        loss = torch.mean(emphasized_loss)

        return loss

def gen_face_with_expression(mu_face):
    E = basis_exp.shape[0]
    exp = torch.tensor(basis_exp)
    ic(exp.size())
    alpha = torch.randn(E,1,1)/8
    face = mu_face + torch.sum(torch.mul(alpha, exp), dim=0)
    ic(face.size())
    return face
    
class VirtDataset(Dataset):
    def __init__(self, cnt=1):
        super().__init__()
        self.cnt = cnt
    def __len__(self):
        return self.cnt
    def __getitem__(self, idx):
        return idx

class AsmModel(pl.LightningModule):
    def __init__(self, K, init_mu_face_rig, base_verts, tgt_verts, inited_gmm_params=None, alpha_mu=1e-5):
        super().__init__()
        J = len(init_mu_face_rig.bones_name)
        self.J = J
        self.K = K 
        self.alpha_mu = alpha_mu
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
        self.tau_scale = nn.Parameter(scale, requires_grad=True)
        # 用 so3 表示旋转
        self.tau_so3 = nn.Parameter(so3, requires_grad=True)
        self.tau_translation = nn.Parameter(translation, requires_grad=True)
        if inited_gmm_params is None or K not in inited_gmm_params.keys():
            pi = torch.rand(J,K)
            mu = (torch.rand(J,K,2)-0.5)/100
            scale_tril = torch.rand(J,K,3)
        else:
            params = inited_gmm_params[K]
            pi = torch.tensor(params["pi"])
            mu = torch.tensor(params["mu"])
            scale_tril = torch.tensor(params["scale_tril"])
        self.pi = nn.Parameter(pi, requires_grad=True)
        self.mu = nn.Parameter(mu, requires_grad=True)
        # sigma = LL^T, L 是一个下三角矩阵, 对角线元素为正数
        self.scale_tril = nn.Parameter(scale_tril, requires_grad=True)
        

    def init_params(self):
        # 获得骨骼初始的 UV 坐标, 以初始化 zeta
        zeta = self.init_state.uv_coords[self.init_state.verts_uv_indices[self.init_state.bones_tail_idx]]
        ic(torch.min(zeta), torch.max(zeta))
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
        mu = (self.mu + self.zeta).view(-1,2) # (J*K, 2)
        tril = self.scale_tril.view(-1,3) # (J*K,3)
        diag = F.softplus(tril[:,:2]) # 对角线大于 0
        t11 = tril[:,[2]]
        zeros = torch.zeros_like(t11)
        # (J*K, 2, 2)
        scale_tril_mat = torch.cat([diag[:,[0]], zeros, 
                                    t11, diag[:,[1]]], dim=-1).view(-1, 2, 2)
        normal2 = MultivariateNormal(mu, scale_tril=scale_tril_mat)
        x = self.init_state.verts_uvs.view(-1,1,2) # vert uvs, (V,1,2)
        log_prob = normal2.log_prob(x) # (V, J*K)
        prob = torch.exp(log_prob).view(-1, self.J, self.K)
        pi = F.softmax(self.pi, dim=-1) # (J, K), normalize pi
        w = torch.mul(pi, prob).sum(dim=-1) # (V, J)
        wg = torch.div(w, torch.sum(w, dim=-1, keepdim=True)) # (V, J)
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
        vt = self.init_state.bones_tail_pos
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
        v_transed = combined_trans.transform_points(v) # (J, V, 3)
        v_weighted = W.T.unsqueeze(-1) * v_transed # (J, V, 3)
        v_updated = torch.sum(v_weighted, dim=0) # (V, 3)
        return v_updated
    
    def save_mesh(self, path):
        src_mesh_path = os.path.join(path, "src.obj")
        tgt_mesh_path = os.path.join(path, "tgt.obj")
        with torch.no_grad():
            src_verts = self.update_verts().detach().cpu().numpy()
            tgt_verts = self.tgt_verts.detach().cpu().numpy()
            tri_indices = self.init_state.tri_indices.detach().cpu().numpy()
            uv_coords = self.init_state.uv_coords.detach().cpu().numpy()
            uv_indices = self.init_state.uv_indices.view(-1).detach().cpu().numpy()
        write_obj_file(
            src_mesh_path,
            src_verts,
            tri_indices,
            uv_coords,
            uv_indices,
        )
        write_obj_file(
            tgt_mesh_path,
            tgt_verts,
            tri_indices,
            uv_coords,
            uv_indices,
        )
    
    def forward(self, x):
        v = self.update_verts()
        return v

    def training_step(self, batch, batch_idx):
        v = self(None)
        loss_verts = F.mse_loss(v, self.tgt_verts)
        # reg_mu_loss = (self.alpha_mu / self.mu.view(-1, 2).size(0)) * torch.sum(self.mu**2)
        loss = loss_verts # + reg_mu_loss
        self.log("loss", loss, prog_bar=True)
        # self.log("reg_mu_loss", reg_mu_loss, prog_bar=True)
        # with torch.no_grad():
        #     dist = torch.cdist(v, self.tgt_verts)
        #     max_dist = torch.max(dist)
        #     self.log("max_dist", max_dist, prog_bar=True)
        return loss

    def configure_optimizers(self):
        default_lr = 1e-3
        params = [
            {"params": self.zeta, "lr": default_lr},
            {"params": self.tau_scale, "lr": default_lr},
            {"params": self.tau_so3, "lr": default_lr},
            {"params": self.tau_translation, "lr": default_lr},
            {"params": self.pi, "lr": default_lr},
            {"params": self.mu, "lr": default_lr},
            {"params": self.scale_tril, "lr": default_lr},
        ]
        optimizer = torch.optim.AdamW(params, lr=default_lr)
        return optimizer

if __name__ == "__main__":
    import pickle
    muface_rig = InitMuFaceRig("data/rig_info.json", "data/mu_face.obj")
    tgt_verts = gen_face_with_expression(muface_rig.verts)
    inited_gmm_params = pickle.load(open("data/inited_gmm_params.pkl", "rb"))
    model = AsmModel(3, muface_rig, base_verts=muface_rig.verts, tgt_verts=tgt_verts, inited_gmm_params=inited_gmm_params)
    Wg = model.gmm_weighting()
    M1 = model.bones_local2world()
    M2 = model.dyn_binding()
    # M = M1.matmul(M2)
    # ic(M)
    # v = model.init_state.verts
    # v_ = model.update_verts()
    # cos_dist = torch.cosine_similarity(v, v_)
    # ic(cos_dist)

