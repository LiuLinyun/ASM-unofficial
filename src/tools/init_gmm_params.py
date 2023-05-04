import os
import pickle
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from icecream import ic

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.append(parent_path)
from model import AsmModel, InitMuFaceRig, VirtDataset



class InitGmmModel(pl.LightningModule):
    def __init__(self, K, mu_face_rig, weights_map):
        super().__init__()
        self.asm_model = AsmModel(K, mu_face_rig, mu_face_rig.verts, mu_face_rig.verts)
        tgt_weights = [weights_map[bone]["weights"] for bone in self.asm_model.init_state.bones_name]
        self.tgt_weights = torch.tensor(tgt_weights).T # (V, J)
        self.params = [
            self.asm_model.pi,
            self.asm_model.mu,
            self.asm_model.scale_tril,
        ]
        self.alpha_mu = 1e-5
        self.lowest_loss = torch.scalar_tensor(torch.inf, dtype=torch.float32)
        self.lowest_loss_params = None

    def update_lowest_loss_params(self, loss):
        if loss < self.lowest_loss:
            self.lowest_loss = loss
            self.lowest_loss_params = [p.clone().detach() for p in self.params]

    def forward(self, x):
        return self.asm_model.gmm_weighting() # (V, J)

    def training_step(self, batch, batch_idx):
        w = self(None)
        reg_mu_loss = self.alpha_mu * torch.norm(self.asm_model.mu)
        loss = F.mse_loss(w, self.tgt_weights) + reg_mu_loss
        self.update_lowest_loss_params(loss)
        self.log("loss", loss, prog_bar=True)
        self.log("reg_mu_loss", reg_mu_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.params, lr=0.05)
        return optimizer

if __name__ == "__main__":
    muface_rig = InitMuFaceRig("data/rig_info.json", "data/mu_face.obj")
    weights_map = pickle.load(open("data/weights_map.pkl", "rb"))
    dataset = VirtDataset(1000)
    data_loader = DataLoader(dataset, batch_size=1)
    inited_params = {}
    for K in range(2,6):
        print(f"init param with K={K}")
        init_gmm_model = InitGmmModel(K, muface_rig, weights_map)
        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(init_gmm_model, train_dataloaders=data_loader)
        inited_params[K] = {
            "pi": init_gmm_model.lowest_loss_params[0].detach().cpu().numpy(),
            "mu": init_gmm_model.lowest_loss_params[1].detach().cpu().numpy(),
            "scale_tril": init_gmm_model.lowest_loss_params[2].detach().cpu().numpy(),
        }
    pickle.dump(inited_params, open("data/inited_gmm_params.pkl", "wb"))
