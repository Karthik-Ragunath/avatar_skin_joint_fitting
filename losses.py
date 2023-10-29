import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	sigmoid_activation = torch.nn.Sigmoid()
	loss = torch.nn.BCELoss()
	output = loss(sigmoid_activation(voxel_src), voxel_tgt)
	return output

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src: b x n_points x 3 
    # point_cloud_tgt: b x n_points x 3  
	loss_chamfer, _ = chamfer_distance(point_cloud_src, point_cloud_tgt)
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian