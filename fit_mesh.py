import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import save_obj
import argparse
import time
from types import SimpleNamespace
from torch.utils.data import SequentialSampler
from torch.utils.data import BatchSampler
from typing import List
from models import MultiFramePoseAE
import torch.optim as optim
import copy
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-f", help="Path to the .obj file", required=True)
    parser.add_argument("--latent_size", "-l_size", help="provide the latent size", type=int, required=False, default=32)
    parser.add_argument("--num_embeddings", "-n_emb", help="provide number of embeddings", type=int, required=False, default=12)
    parser.add_argument("--num_experts", "-n_exp", help="provide number of experts", type=int, required=False, default=6)
    parser.add_argument("--num_frames", "-n_frames", help="provide number of frames", type=int, required=False, default=60)
    parser.add_argument("--load_saved_model", "-load_model", help="if saved model must be loaded", action='store_true')
    parser.add_argument("--num_condition_frames", "-n_cond", help="provide number of condition frames", type=int, required=False, default=1)
    parser.add_argument("--epoch_save_interval", "-save_interval", help="provide the eppch interval to save the model", type=int, required=True, default=10)
    args = parser.parse_args()
    return args

def load_mesh(file_path):
    # Load the mesh from .obj file
    mesh = load_objs_as_meshes([file_path], device=torch.device("cpu"))
    return mesh

def feed_auto_encoder(pose_auto_encoder: MultiFramePoseAE, ground_truth: torch.Tensor, condition: torch.Tensor, future_weights: torch.Tensor):
    condition = condition.flatten(start_dim=1, end_dim=2)
    flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2)
    output_shape = (-1, 1, pose_auto_encoder.frame_size)

    vae_output, mu_prior, logvar_prior = pose_auto_encoder(condition)
    vae_output = vae_output.view(output_shape)
    recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1))
    recon_loss = recon_loss.mul(future_weights).sum()

    return (vae_output, mu_prior, logvar_prior), (recon_loss)

def load_mesh(file_path: str):
    mesh = load_mesh(file_path)
    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    return mesh, vertices, faces

def main(args: SimpleNamespace, source_mesh_file_paths: List, target_mesh_file_paths: List):
    frame_size = args.frame_size
    num_epochs = args.num_epochs
    latent_size = args.latent_size
    num_condition_frames = args.num_condition_frames
    num_experts = args.num_experts
    epoch_save_interval = args.epoch_save_interval

    future_weights = (
        torch.ones(args.num_future_predictions)
        .to(args.device)
        .div_(args.num_future_predictions)
    )

    pose_auto_encoder = MultiFramePoseAE(
        frame_size,
        latent_size,
        num_condition_frames,
        num_experts,
    ).to(args.device)

    vae_optimizer = optim.Adam(pose_auto_encoder.parameters(), lr=args.initial_lr)

    # buffer for later
    shape = (args.mini_batch_size, args.num_condition_frames, frame_size)
    condition = torch.empty(shape).to(args.device)
    ground_truth = torch.empty(shape).to(args.device)
    for epoch in range(1, num_epochs + 1):
        ep_recon_loss = 0
        mesh_index = 1 # for maintaining coding practice, since mesh_index will be used down the line for division
        for mesh_index, (source_mesh_file_path, target_mesh_file_path) in enumerate(zip(source_mesh_file_paths, target_mesh_file_paths)):
            source_mesh, source_vertices, source_faces = load_mesh(source_mesh_file_path)
            target_mesh, target_vertices, target_faces = load_mesh(target_mesh_file_path)
            selectable_indices = list(range(args.num_condition_frames, source_mesh.shape[0], 1))
            sampler = BatchSampler(
                SequentialSampler(selectable_indices),
                args.mini_batch_size,
                drop_last=True,
            )
            mini_batch_index = 1 # for maintaining coding practice, since mini_batch_index will be used down the line for division
            for mini_batch_index, indices in enumerate(sampler):
                t_indices = torch.LongTensor(indices)
                condition_range = (
                    t_indices.repeat((args.num_condition_frames, 1)).t()
                    - torch.arange(args.num_condition_frames - 1, -1, -1).long()
                )
                condition[:, :args.num_condition_frames].copy_(source_vertices[condition_range])
                ground_truth[:, :args.num_condition_frames].copy_(target_vertices[condition_range])
                (vae_output, _, _), (recon_loss) = feed_auto_encoder(
                    pose_auto_encoder, ground_truth, condition, future_weights
                )
                vae_optimizer.zero_grad()
                (recon_loss).backward()
                vae_optimizer.step()

                ep_recon_loss += float(recon_loss)
        
        avg_ep_recon_loss = ep_recon_loss / (mini_batch_index * mesh_index)
        print(f"Average Reconstruction for epoch: {epoch} is {avg_ep_recon_loss}")
        if epoch % epoch_save_interval == 0:
            os.makedirs("save_model", exist_ok=True)
            torch.save(copy.deepcopy(pose_auto_encoder).cpu(), os.path.join("save_model", f"{epoch}.pt"))


if __name__ == "__main__":
    args = parse_arguments()
    # setup parameters
    args.num_epochs = 100
    args.mini_batch_size = 64
    args.initial_lr = 1e-4
    args.final_lr = 1e-7
    main(args)
