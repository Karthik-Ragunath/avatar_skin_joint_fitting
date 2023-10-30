import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import save_obj
import argparse
import time
from types import SimpleNamespace
from torch.utils.data import SequentialSampler
from torch.utils.data import BatchSampler
from typing import List
from models import MultiFramePoseWithJointsAE
import torch.optim as optim
import copy
import os
import glob
import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_files_dir", "-src", help="Path to the source .obj files", required=True)
    parser.add_argument("--target_files_dir", "-tar", help="Path to the target .obj files", required=False)
    parser.add_argument("--joint_files_dir", "-src_joint", help="Path to the joint .obj files", required=True)
    parser.add_argument("--latent_size", "-l_size", help="provide the latent size", type=int, required=False, default=32)
    parser.add_argument("--num_embeddings", "-n_emb", help="provide number of embeddings", type=int, required=False, default=12)
    parser.add_argument("--num_experts", "-n_exp", help="provide number of experts", type=int, required=False, default=6)
    parser.add_argument("--num_frames", "-n_frames", help="provide number of frames", type=int, required=False, default=60)
    parser.add_argument("--num_condition_frames", "-n_cond", help="provide number of condition frames", type=int, required=False, default=1)
    parser.add_argument("--device", "-dev", help="provide the device to use for training: cuda:0 or cpu", type=str, required=False, default="cuda:0")
    parser.add_argument("--model_type", "-m_type", help="provide the type of the trained model used", type=str, required=False, default="with_joints")
    parser.add_argument("--model_saved_path", "-saved_path", help="provide the directory where the trained model is saved", type=str, required=True)
    parser.add_argument("--is_target_available", "-tar_avail", help="is target data available?", action="store_true")
    args = parser.parse_args()
    return args

def load_mesh(file_path):
    # Load the mesh from .obj file
    mesh = load_objs_as_meshes([file_path], device=torch.device("cpu"))
    return mesh

def load_mesh_data(file_path: str):
    mesh = load_mesh(file_path)
    vertices = mesh.verts_list()[0]
    faces = mesh.faces_list()[0]
    return mesh, vertices, faces

def load_joint_mesh(file_path: str):
    mesh = load_mesh(file_path)
    vertices = mesh.verts_list()[0]
    return mesh, vertices, None

def feed_auto_encoder(
    pose_auto_encoder: MultiFramePoseWithJointsAE, 
    ground_truth: torch.Tensor, 
    condition: torch.Tensor, 
    future_weights: torch.Tensor,
    joint_condition: torch.Tensor
):
    condition_flattened = condition.flatten(start_dim=1, end_dim=2) # torch.Size([args.mini_batch_size, 9])
    flattened_truth = ground_truth.flatten(start_dim=1, end_dim=2) # torch.Size([args.mini_batch_size, 3])
    joint_condition_flattened = joint_condition.clone().flatten(start_dim=1, end_dim=2) # torch.Size([args.mini_batch_size, 249])
    output_shape = (-1, 1, pose_auto_encoder.frame_size) # (-1, 1, 3)

    vae_output, mu_prior, logvar_prior = pose_auto_encoder(condition_flattened, joint_condition_flattened) # torch.Size([args.mini_batch_size, 3]), torch.Size([args.mini_batch_size, 32]), torch.Size([args.mini_batch_size, 32])
    vae_output = vae_output.view(output_shape) # torch.Size([args.mini_batch_size, 1, 3])
    recon_loss = (vae_output - ground_truth).pow(2).mean(dim=(0, -1)) # torch.Size([3])
    recon_loss = recon_loss.mul(future_weights).sum() # tensor(1.9944, device='cuda:0', grad_fn=<SumBackward0>)

    return (vae_output, mu_prior, logvar_prior), (recon_loss)

def feed_auto_encoder_inference(
    pose_auto_encoder: MultiFramePoseWithJointsAE, 
    condition: torch.Tensor, 
    joint_condition: torch.Tensor
):
    condition_flattened = condition.flatten(start_dim=1, end_dim=2) # torch.Size([args.mini_batch_size, 9])
    joint_condition_flattened = joint_condition.clone().flatten(start_dim=1, end_dim=2) # torch.Size([args.mini_batch_size, 249])
    output_shape = (-1, 1, pose_auto_encoder.frame_size) # (-1, 1, 3)
    vae_output, mu_prior, logvar_prior = pose_auto_encoder(condition_flattened, joint_condition_flattened) # torch.Size([args.mini_batch_size, 3]), torch.Size([args.mini_batch_size, 32]), torch.Size([args.mini_batch_size, 32])
    vae_output = vae_output.view(output_shape) # torch.Size([args.mini_batch_size, 1, 3])
    return (vae_output, mu_prior, logvar_prior), (None)

def main(args: SimpleNamespace, source_mesh_file_paths: List, target_mesh_file_paths: List, joints_file_paths: List):
    frame_size = args.frame_size
    num_epochs = args.num_epochs
    latent_size = args.latent_size
    num_condition_frames = args.num_condition_frames
    num_experts = args.num_experts
    epoch_save_interval = args.epoch_save_interval
    num_joints = args.num_joints

    future_weights = (
        torch.ones(1)
        .to(args.device)
        .div_(1)
    )

    pose_auto_encoder = MultiFramePoseWithJointsAE(
        frame_size,
        latent_size,
        num_condition_frames,
        num_experts,
        num_joints
    ).to(args.device)
    pose_auto_encoder.eval()

    # buffer for later
    # joint_shape = (args.mini_batch_size, num_joints, frame_size)
    shape = (args.mini_batch_size, args.num_condition_frames, frame_size)
    condition = torch.empty(shape).to(args.device)
    ground_truth = torch.empty(shape).to(args.device)
    for mesh_index, (source_mesh_file_path, joint_file_path) in enumerate(zip(source_mesh_file_paths, joints_file_paths)):
        if args.is_target_available:
            target_mesh_file_path = target_mesh_file_paths[mesh_index]
        mesh_reconstruction_loss = 0
        source_mesh, source_vertices, source_faces = load_mesh_data(source_mesh_file_path)
        joint_mesh, joint_vertices, _ = load_joint_mesh(joint_file_path) 
        if args.is_target_available:
            target_mesh, target_vertices, target_faces = load_mesh_data(target_mesh_file_path)
        joint_condition = joint_vertices.clone().unsqueeze(0).repeat(args.mini_batch_size, 1, 1).to(args.device) # torch.Size([16364, 83, 3])
        selectable_indices = range(args.num_condition_frames - 1, source_vertices.shape[0], 1)
        sampler = BatchSampler(
            SequentialSampler(selectable_indices),
            args.mini_batch_size,
            drop_last=True,
        )
        predicted_pose_list = []
        mini_batch_index = 1 # for maintaining coding practice, since mini_batch_index will be used down the line for division
        for mini_batch_index, indices in enumerate(sampler):
            with torch.no_grad():
                selected_indices = [selectable_indices[index] for index in indices]
                t_indices = torch.LongTensor(selected_indices) # torch.Size([args.mini_batch_size])
                condition_range = (
                    t_indices.repeat((args.num_condition_frames, 1)).t()
                    - torch.arange(args.num_condition_frames - 1, -1, -1).long()
                ) # torch.Size([args.mini_batch_size, 3])
                offset = 0 # offset set to zero, increasing it will modify which future frame you want to predict
                prediction_range = (
                    t_indices.repeat((1, 1)).t()
                    + torch.arange(offset, offset + 1).long()
                ) # torch.Size([args.mini_batch_size, 1])
                if args.is_target_available:
                    condition[:, :args.num_condition_frames].copy_(source_vertices[condition_range]) # torch.Size([args.mini_batch_size, 3, 3])
                    ground_truth = target_vertices[prediction_range].to(args.device) # torch.Size([args.mini_batch_size, 1, 3])
                    (vae_output, _, _), (recon_loss) = feed_auto_encoder(
                        pose_auto_encoder, ground_truth.clone(), condition.clone(), future_weights, joint_condition.clone()
                    ) # torch.Size([args.mini_batch_size, 1, 3])
                    mesh_reconstruction_loss += float(recon_loss)
                    logger.info(f'mesh_index: {mesh_index}; mini_batch_index: {mini_batch_index} - error: {float(recon_loss)}')
                else:
                    (vae_output, _, _), (_) = feed_auto_encoder_inference(
                        pose_auto_encoder, condition.clone(), joint_condition.clone()
                    ) # torch.Size([args.mini_batch_size, 1, 3])
                predicted_pose_list.append(vae_output[:, 0].clone().detach().cpu())
        if args.is_target_available:
            avg_mesh_recon_loss = mesh_reconstruction_loss / ((mini_batch_index + 1) * (mesh_index + 1))
            logger.info(f"Average Reconstruction for mesh: {source_mesh_file_path} is {avg_mesh_recon_loss}")
        predicted_pose_tensor = torch.stack(predicted_pose_list, dim=0)
        copy_pose_list = []
        for i in range(0, args.num_condition_frames-1):
            copy_pose_list.append(source_vertices[i])
        if copy_pose_list:
            copy_pose_tensor = torch.stack(copy_pose_list, dim=0)
            predicted_pose_tensor = torch.cat((copy_pose_tensor, predicted_pose_tensor), dim=0)            
        os.makedirs(os.path.join('inference_meshes', args.model_type), exist_ok=True)
        save_obj(f=os.path.join('inference_meshes', args.model_type, source_mesh_file_path.split('/')[-1]), verts=predicted_pose_tensor, faces=source_faces)


if __name__ == "__main__":
    timestamp = time.time()
    human_readable_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(timestamp))
    args = parse_arguments()
    # setup parameters
    args.num_epochs = 2
    args.mini_batch_size = 32729
    args.initial_lr = 1e-4
    args.final_lr = 1e-7
    args.frame_size = 3 # vertex data dimension in 3d world
    args.timestamp = human_readable_time
    args.num_joints = 83
    source_files_dir = args.source_files_dir
    source_files = sorted(glob.glob(os.path.join(source_files_dir, "*.obj")))
    joint_files_dir = args.joint_files_dir
    joint_files = sorted(glob.glob(os.path.join(joint_files_dir, "*.obj")))
    if args.is_target_available:
        target_files_dir = args.target_files_dir
        target_files = sorted(glob.glob(os.path.join(target_files_dir, "*.obj")))
    else:
        target_files = None
    os.makedirs(os.path.join('logs', args.model_type), exist_ok=True)
    file_handler = logging.FileHandler(os.path.join('logs', args.model_type, args.timestamp))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    main(args=args, source_mesh_file_paths=source_files,target_mesh_file_paths=target_files, joints_file_paths=joint_files)
