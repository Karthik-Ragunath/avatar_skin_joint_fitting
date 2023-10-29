import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import save_obj
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", help="Path to the .obj file", required=True)
    args = parser.parse_args()
    return args

def load_mesh(file_path):
    # Load the mesh from .obj file
    mesh = load_objs_as_meshes([file_path], device=torch.device("cpu"))
    return mesh

if __name__ == "__main__":
    args = parse_arguments()
    mesh = load_mesh(args.file_path)
    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    # print(mesh)
    save_obj('pytorch_3d_mesh.obj', vertices, faces)
