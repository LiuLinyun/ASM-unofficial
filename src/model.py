import os
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Uniform
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch3d import transforms as trans

from tools.init_mu_face_rig import InitMuFaceRig
from tools.mesh_utils import MeshUVProjecter
from tools.face_dataset import basis_exp
from tools.read_write_obj import write_obj_file

from icecream import ic

def softweight(x, dim=-1):
    # 先对所有的数做softplus
    x = F.softplus(x)
    sum = torch.sum(x, dim=dim, keepdim=True)
    return torch.div(x, sum)

def gen_face_with_expression(mu_face):
    E = basis_exp.shape[0]
    exp = torch.tensor(basis_exp)
    alpha = torch.randn(E,1,1)/8
    face = mu_face + torch.sum(torch.mul(alpha, exp), dim=0)
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
    def __init__(self, K, init_mu_face_rig, base_verts, tgt_verts, inited_gmm_params=None, lr=1e-3, save_obj=False):
        super().__init__()
        J = len(init_mu_face_rig.bones_name)
        self.J = J
        self.K = K 
        self.lr = lr
        self.save_obj = save_obj
        self.init_state = init_mu_face_rig
        self.mesh_uv_projecter = MeshUVProjecter(
            self.init_state.verts,
            self.init_state.tri_indices,
            self.init_state.uv_coords,
            self.init_state.uv_indices
        )
        # 用到的数据
        self.verts_uvs = nn.Parameter(self.init_state.verts_uvs, requires_grad=False)
        self.M_root_local2obj = nn.Parameter(self.init_state.bones_dict["face"].M_local2obj, requires_grad=False)
        self.bones_tail_pos = nn.Parameter(self.init_state.bones_tail_pos, requires_grad=False)
        self.bones_M_local2obj = nn.Parameter(self.init_state.bones_M_local2obj, requires_grad=False)
        self.base_verts = nn.Parameter(base_verts, requires_grad=False)
        self.tgt_verts = nn.Parameter(tgt_verts, requires_grad=False)
        # 可学习的参数
        self.zeta = nn.Parameter(self.init_state.bones_uvs.view(-1,1,2), requires_grad=True) # (J,1,2)
        scale, so3, translation = self.init_tau_params()
        self.tau_scale = nn.Parameter(scale, requires_grad=True) # (J,3)
        # 用 so3 表示旋转
        self.tau_so3 = nn.Parameter(so3, requires_grad=True)
        self.tau_translation = nn.Parameter(translation, requires_grad=True)
        if inited_gmm_params is None:
            pi = Uniform(0, 10).sample((J,K))
            mu = Uniform(-0.1, 0.1).sample((J,K,2))
            scale_tril = Uniform(-1, 1).sample((J,K,3))
        else:
            params = inited_gmm_params
            pi = torch.tensor(params["pi"])
            mu = torch.tensor(params["mu"])
            scale_tril = torch.tensor(params["scale_tril"])
        self.pi = nn.Parameter(pi, requires_grad=True)
        self.mu = nn.Parameter(mu, requires_grad=True)
        # sigma = LL^T, L 是一个下三角矩阵, 对角线元素为正数
        self.scale_tril = nn.Parameter(scale_tril, requires_grad=True)

    def init_tau_params(self):
        # 将所有骨骼的父骨骼更改为根骨骼(头部内部)
        # M_tau = M_local2root = M_obj2root M_local2obj = M_root2obj^{-1} M_local2obj
        # M_root2obj = self.M_root_local2obj
        # M_local2obj = self.bones_M_local2obj
        M_local2root = self.M_root_local2obj.inverse().matmul(self.bones_M_local2obj)

        # 分解变换矩阵, 得到位移, 旋转, 缩放
        sR = M_local2root[:,:3,:3]
        s = torch.norm(sR, dim=1, keepdim=True)
        R = sR / s
        so3 = trans.so3_log_map(R.transpose(-1,-2))
        T = M_local2root[:,:3,3]
        return s.view(-1,3), so3, T
    
    def gmm_weighting(self):
        mu = (self.mu + self.zeta).view(-1,2) # (J*K, 2)
        tril = self.scale_tril.view(-1,3) # (J*K,3)
        diag = F.softplus(tril[:,:2]) # 对角线大于 0
        t11 = tril[:,[2]]
        zeros = torch.zeros_like(t11)
        scale_tril_mat = torch.cat([diag[:,[0]], zeros, 
                                    t11, diag[:,[1]]], dim=-1).view(-1, 2, 2)
        normal2 = MultivariateNormal(mu, scale_tril=scale_tril_mat)
        x = self.verts_uvs.view(-1,1,2) # vert uvs, (V,1,2)
        log_prob = normal2.log_prob(x) # (V, J*K)
        prob = torch.exp(log_prob).view(-1, self.J, self.K)
        pi = softweight(self.pi, dim=-1) # (J, K)
        w = torch.mul(pi, prob).sum(dim=-1) # (V, J)
        wg = softweight(w, dim=-1)
        return wg
    
    def bones_local2world(self):
        trans_tau = trans.Scale(self.tau_scale).compose(
            trans.Rotate(trans.so3_exp_map(self.tau_so3))).compose(
            trans.Translate(self.tau_translation))
        M_tau = trans_tau.get_matrix().transpose(-1,-2)
        M_l2w = torch.matmul(self.M_root_local2obj, M_tau)
        return M_l2w
        
    
    def dyn_binding(self):
        # psi = self.mesh_uv_projecter.uv2mesh(self.zeta) - self.vt + self.psi_init
        # 但是实际计算只需要骨骼顶点 psi 的变化量 delta_psi
        # 更新的绑定姿态矩阵 B' = TB
        # T 为世界坐标系下骨骼的变换(这里只有平移)
        # B 就是骨骼原绑定姿态的local2world matrix
        # (J, 3)
        delta_psi = self.mesh_uv_projecter.uv2mesh(self.zeta.view(-1,2)) - self.bones_tail_pos
        # (J,4,4)
        B = self.bones_M_local2obj
        # 添加平移 
        # B[:,:3,3] += delta_psi
        B_ = torch.empty_like(B)
        B_[:,:,:3] = B[:,:,:3]
        B_[:,:,3] = B[:,:,3]
        B_[:,:3,3] = B[:,:3,3] + delta_psi
        B_inv = B_.inverse()
        return B_inv

    def forward(self, x):
        W = self.gmm_weighting() # (V, J)
        M_l2w = self.bones_local2world() # (J, 4, 4)
        B_inv = self.dyn_binding() # (J, 4, 4)
        combined_trans = trans.Transform3d(matrix=torch.matmul(M_l2w,B_inv).transpose(-1,-2))
        v_transed = combined_trans.transform_points(self.base_verts) # (J, V, 3)
        v_weighted = W.T.unsqueeze(-1) * v_transed # (J, V, 3)
        v_updated = torch.sum(v_weighted, dim=0) # (V, 3)
        return v_updated
    
    def save_mesh(self, path):
        with torch.no_grad():
            src_verts = self(None).detach().cpu().numpy()
            tri_indices = self.init_state.tri_indices.detach().cpu().numpy()
            uv_coords = self.init_state.uv_coords.detach().cpu().numpy()
            uv_indices = self.init_state.uv_indices.view(-1).detach().cpu().numpy()
            write_obj_file(path, src_verts, tri_indices, uv_coords, uv_indices)

    def training_step(self, batch, batch_idx):
        v = self(None)
        loss = 1e4*F.mse_loss(v, self.tgt_verts)
        self.log("loss", loss, prog_bar=True)
        if self.save_obj and batch_idx%100 == 0:
            self.save_mesh(os.path.join(self.logger.log_dir, f"step_{batch_idx}.obj"))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

if __name__ == "__main__":
    import pickle
    muface_rig = InitMuFaceRig("data/rig_info.json", "data/mu_face.obj")
    tgt_verts = gen_face_with_expression(muface_rig.verts)
    inited_gmm_params = pickle.load(open("data/inited_gmm_params_K3.pkl", "rb"))
    model = AsmModel(3, muface_rig, base_verts=muface_rig.verts, tgt_verts=tgt_verts, inited_gmm_params=inited_gmm_params)
    Wg = model.gmm_weighting()
    M1 = model.bones_local2world()
    M2 = model.dyn_binding()
    M = M1.matmul(M2)
    ic(M)
    # v = model.init_state.verts
    # v_ = model.update_verts()
    # cos_dist = torch.cosine_similarity(v, v_)
    # ic(cos_dist)

