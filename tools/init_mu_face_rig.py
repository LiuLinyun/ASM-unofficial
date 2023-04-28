import json
import numpy as np
import torch
from pytorch3d import transforms as trans
from tools.read_write_obj import read_obj_file

class BoneNode():
    def __init__(
        self,
        head_idx,
        tail_idx,
        head_pos,
        tail_pos,
        M_local2obj,
        parent,
        children,
    ):
        self.head_idx = head_idx
        self.tail_idx = tail_idx
        self.head_pos = head_pos
        self.tail_pos = tail_pos
        self.M_local2obj = M_local2obj
        self.parent = parent
        self.children = children

class InitMuFaceRig():
    def __init__(self, rig_info_file, mu_face_file):
        (
            self.verts, 
            self.tri_indices, 
            self.uv_coords, 
            self.uv_indices,
            self.verts_uv_indices,
        ) = self.read_mu_face(mu_face_file)
        (
            self.bones_dict
        ) = self.read_rig(rig_info_file, self.verts)
        # 根骨骼 "face" 不参与优化
        self.bones_name = [k for k in self.bones_dict.keys() if k != "face"]
        self.bones_name_idx = dict([(k, i) for (i,k) in enumerate(self.bones_name)])
        self.bones_list = [self.bones_dict[k] for k in self.bones_name]
        self.bones_head_idx = [self.bones_dict[k].head_idx for k in self.bones_name]
        self.bones_tail_idx = [self.bones_dict[k].tail_idx for k in self.bones_name]
        self.bones_head_pos = torch.cat([self.bones_dict[k].head_pos for k in self.bones_name], dim=0)
        self.bones_tail_pos = torch.cat([self.bones_dict[k].tail_pos for k in self.bones_name], dim=0)
        self.bones_M_local2obj = torch.cat([self.bones_dict[k].M_local2obj.unsqueeze(0) for k in self.bones_name], dim=0)

    def read_mu_face(self, mu_face_file):
        verts, tri_indices, uv_coords, uv_indices = read_obj_file(mu_face_file)
        uv_indices = uv_indices.reshape(-1,3)
        verts_uv_indices = [None]*verts.shape[0]
        for tri_idx in range(tri_indices.shape[0]):
            id_abc = tri_indices[tri_idx]
            uv_abc = uv_indices[tri_idx]
            for i in range(3):
                if verts_uv_indices[id_abc[i]] is None:
                    verts_uv_indices[id_abc[i]] = uv_abc[i]
        for idx in verts_uv_indices:
            if idx is None:
                print("WARN: there are some vertices not unprojected to UV space!")
        return (
            torch.tensor(verts, dtype=torch.float32),
            torch.tensor(tri_indices, dtype=torch.int32),
            torch.tensor(uv_coords, dtype=torch.float32),
            torch.tensor(uv_indices, dtype=torch.int32),
            torch.tensor(verts_uv_indices, dtype=torch.int32),
        )

    def read_rig(self, rig_info_file, verts):
        rig_info = json.load(open(rig_info_file, "r"))

        mu_face_info = rig_info["mu_face"]
        # 从 rig_file_info 的世界空间转到对象空间的变换矩阵
        M_world2obj = torch.tensor(mu_face_info["matrix_world"], dtype=torch.float32).inverse()

        # 从所有骨骼的对象空间到世界空间的变换矩阵(FIXME later .T)
        M_rig2world = torch.tensor(rig_info["metarig"]["matrix_world"], dtype=torch.float32)
        bones_info = rig_info["metarig"]["bones"]
        
        # 获取所有骨骼顶点的位置
        bones_dict = {}
        M_rig2obj = torch.matmul(M_world2obj, M_rig2world)
        trans2obj = trans.Transform3d(matrix=M_rig2obj.T)
        for name, bone in bones_info.items():
            head = torch.tensor(bone["head"], dtype=torch.float32)
            tail = torch.tensor(bone["tail"], dtype=torch.float32)
            head = trans2obj.transform_points(head.view(1,3))
            tail = trans2obj.transform_points(tail.view(1,3))
            nearest_face_head_idx = torch.argmin(torch.pairwise_distance(head, verts))
            nearest_face_tail_idx = torch.argmin(torch.pairwise_distance(tail, verts))
            M_local2rig = torch.tensor(bone["matrix"], dtype=torch.float32)

            bones_dict[name] = BoneNode(
                head_idx = nearest_face_head_idx.item(),
                tail_idx = nearest_face_tail_idx.item(),
                head_pos = head,
                tail_pos = tail,
                M_local2obj = torch.matmul(M_rig2obj, M_local2rig),
                parent = bone["parent"],
                children = bone["children"],
            )
        return bones_dict


if __name__ == "__main__":
    import plotly.express as px
    from icecream import ic

    init = InitMuFaceRig("data/rig_info.json", "data/mu_face.obj")
    verts = init.verts
    indices = [b.tail_idx for b in init.bones_dict.values()]
    verts_ = verts[indices].numpy()
    # fig = px.scatter_3d(x=verts_[:,0], y=verts_[:,1], z=verts_[:,2])
    # fig.show()



