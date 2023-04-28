import numpy as np

# 读取.obj文件
def read_obj_file(filename):
    vertices = []
    faces = []
    texture_coords = []
    texture_indices = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = line.split()[1:]
                vertices.append(list(map(float, vertex)))
            elif line.startswith('f '):
                face = line.split()[1:]
                vertex_indices = []
                for i in face:
                    indices = i.split('/')
                    vertex_indices.append(int(indices[0]))
                    texture_indices.append(int(indices[1]))
                faces.append(vertex_indices)
            elif line.startswith('vt '):
                texture_coord = line.split()[1:]
                texture_coords.append(list(map(float, texture_coord)))

    return np.array(vertices), np.array(faces)-1, np.array(texture_coords), np.array(texture_indices)-1

# 写入.obj文件
def write_obj_file(filename, vertices, faces, texture_coords, texture_indices):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for vt in texture_coords:
            f.write(f"vt {vt[0]} {vt[1]}\n")
        for i, face in enumerate(faces):
            f.write("f")
            for j in range(len(face)):
                f.write(f" {face[j]+1}/{texture_indices[i*len(face)+j]+1}")
            f.write("\n")

if __name__ == "__main__":
    # 示例用法
    # 创建一个简单的正方体模型
    vertices = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    faces = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [1, 2, 6],
        [1, 5, 6],
        [1, 3, 7],
        [1, 5, 7],
        [2, 4, 8],
        [2, 6, 8],
        [3, 4, 8],
        [3, 7, 8],
        [5, 6, 8],
        [5, 7, 8]
    ])
    texture_coords = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    texture_indices = np.array([
        1, 2, 3, 2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3, 2, 3, 4,
        1, 2, 4, 1, 3, 4, 1, 2, 3, 2, 3, 4, 1, 2, 4, 1, 3, 4
    ])

    # 写入.obj文件
    write_obj_file('data/example.obj', vertices, faces, texture_coords, texture_indices)

    # 读取.obj文件
    read_vertices, read_faces, read_texture_coords, read_texture_indices = read_obj_file('data/example.obj')

    # 打印读取的数据
    print("读取的顶点：")
    print(read_vertices)
    print("读取的面：")
    print(read_faces)
    print("读取的纹理坐标：")
    print(read_texture_coords)
    print("读取的纹理索引：")
    print(read_texture_indices)
