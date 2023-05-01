import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import sys
sys.path.append('.')
from tools.init_mu_face_rig import InitMuFaceRig
from model import gen_face_with_expression, VirtDataset, AsmModel

muface_rig = InitMuFaceRig("data/rig_info.json", "data/mu_face.obj")
tgt_verts = gen_face_with_expression(muface_rig.verts)
dataset = VirtDataset()
data_loader = DataLoader(dataset, batch_size=1)
model = AsmModel(15, muface_rig, base_verts=muface_rig.verts, tgt_verts=tgt_verts)
trainer = pl.Trainer(max_epochs=1000)
trainer.fit(model, train_dataloaders=data_loader)