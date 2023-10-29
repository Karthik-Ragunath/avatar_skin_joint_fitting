import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import save_obj
import argparse
import time
from types import SimpleNamespace
from torch.utils.data import SequentialSampler
from torch.utils.data import BatchSampler
from typing import List

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", help="Path to the .obj file", required=True)
    parser.add_argument("--latent_size", "-l_size", help="provide the latent size", type=int, required=False, default=32)
    parser.add_argument("--num_embeddings", "-n_emb", help="provide number of embeddings", type=int, required=False, default=12)
    parser.add_argument("--num_experts", "-n_exp", help="provide number of experts", type=int, required=False, default=6)
    parser.add_argument("--num_frames", "-n_frames", help="provide number of frames", type=int, required=False, default=60)
    parser.add_argument("--load_saved_model", "-load_model", help="if saved model must be loaded", action='store_true')
    parser.add_argument("--num_condition_frames", "-n_cond", help="provide number of condition frames", type=int, required=False, default=1)
    args = parser.parse_args()
    return args

def load_mesh(file_path):
    # Load the mesh from .obj file
    mesh = load_objs_as_meshes([file_path], device=torch.device("cpu"))
    return mesh

def fit_mesh(mesh_src, mesh_tgt, args):
    start_iter = 0
    start_time = time.time()

    deform_vertices_src = torch.zeros(mesh_src.verts_packed().shape, requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([deform_vertices_src], lr = args.lr)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        new_mesh_src = mesh_src.offset_verts(deform_vertices_src)

        sample_trg = sample_points_from_meshes(mesh_tgt, args.n_points)
        sample_src = sample_points_from_meshes(new_mesh_src, args.n_points)

        loss_reg = losses.chamfer_loss(sample_src, sample_trg)
        loss_smooth = losses.smoothness_loss(new_mesh_src)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        print("[%4d/%4d]; ttime: %.0f (%.2f); loss: %.3f" % (step, args.max_iter, total_time,  iter_time, loss_vis))        
    
    mesh_src.offset_verts_(deform_vertices_src)
    print('Done!')

def load_mesh(file_path: str):
    mesh = load_mesh(file_path)
    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    return mesh, vertices, faces

def main(args: SimpleNamespace, source_mesh_file_paths: List, target_mesh_file_paths: List):
    frame_size = args.frame_size
    num_epochs = args.num_epochs
    num_condition_frames = args.num_condition_frames

    future_weights = (
        torch.ones(args.num_future_predictions)
        .to(args.device)
        .div_(args.num_future_predictions)
    )


    # buffer for later
    shape = (args.mini_batch_size, args.num_condition_frames, frame_size)
    history = torch.empty(shape).to(args.device)
    for epoch in range(1, num_epochs + 1):
        for source_mesh_file_path, target_mesh_file_path in zip(source_mesh_file_paths, target_mesh_file_paths):
            source_mesh, source_vertices, source_faces = load_mesh(source_mesh_file_path)
            target_mesh, target_vertices, target_faces = load_mesh(target_mesh_file_path)
            selectable_indices = list(range(args.num_condition_frames, source_mesh.shape[0], 1))
            sampler = BatchSampler(
                SequentialSampler(selectable_indices),
                args.mini_batch_size,
                drop_last=True,
            )

if __name__ == "__main__":
    args = parse_arguments()
    # setup parameters
    mesh = load_mesh(args.file_path)
    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    # print(mesh)
    save_obj('pytorch_3d_mesh.obj', vertices, faces)
