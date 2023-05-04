import pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from tools.read_write_obj import read_obj_file
from tools.init_mu_face_rig import InitMuFaceRig
from model import gen_face_with_expression, VirtDataset, AsmModel

muface_rig = InitMuFaceRig("data/rig_info.json", "data/mu_face.obj")
inited_gmm_params = pickle.load(open("data/inited_gmm_params.pkl", "rb"))
# tgt_verts = gen_face_with_expression(muface_rig.verts)
tgt_verts, _, _, _, = read_obj_file("data/tgt_example.obj")
tgt_verts = torch.tensor(tgt_verts, dtype=torch.float32)
dataset = VirtDataset(100)
data_loader = DataLoader(dataset, batch_size=1)
model = AsmModel(3, muface_rig, base_verts=muface_rig.verts, tgt_verts=tgt_verts, inited_gmm_params=inited_gmm_params)
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_dataloaders=data_loader)
model.save_mesh("data/output")
