import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshDeformEncoder(torch.nn):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        prior_input_size = frame_size * num_condition_frames
        self.fc1_prior = nn.Linear(prior_input_size, hidden_size)
        self.fc2_prior = nn.Linear(prior_input_size + hidden_size, hidden_size)
        self.mu_prior = nn.Linear(prior_input_size + hidden_size, latent_size)
        self.logvar_prior = nn.Linear(prior_input_size + hidden_size, latent_size)
        
    def encode_prior(self, c):
        h1 = F.elu(self.fc1_prior(c, dim=1))
        h2 = F.elu(self.fc2_prior(torch.cat((c, h1), dim=1)))
        s = torch.cat((c, h2), dim=1)
        return self.mu_prior(s), self.logvar_prior(s)

    def forward(self, c):
        mu_prior, logvar_prior = self.encode_prior(c)
        z = mu_prior + logvar_prior
        return z, mu_prior, logvar_prior
    
class MeshDeformerDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = 1 * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c

        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out

class MultiFramePoseAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_experts
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames

        hidden_size = 256
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
        )
        self.encoder = MeshDeformEncoder(*args)
        self.decoder = MeshDeformerDecoder(*args, num_experts)

    def encode(self, c):
        _, mu_prior, logvar_prior = self.encoder(c)
        return mu_prior, logvar_prior

    def forward(self, c):
        z, mu_prior, logvar_prior = self.encoder(c)
        return self.decoder(z, c), mu_prior, logvar_prior

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)


class MeshDeformWithJointsEncoder(torch.nn):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_joints
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        prior_input_size = frame_size * num_condition_frames
        prior_joint_size = frame_size * num_joints
        self.fc1_prior = nn.Linear(prior_input_size + prior_joint_size, hidden_size)
        self.fc2_prior = nn.Linear(prior_input_size + prior_joint_size + hidden_size, hidden_size)
        self.mu_prior = nn.Linear(prior_input_size + prior_joint_size + hidden_size, latent_size)
        self.logvar_prior = nn.Linear(prior_input_size + prior_joint_size + hidden_size, latent_size)
        
    def encode_prior(self, c, joints):
        h1 = F.elu(self.fc1_prior(torch.cat(c, joints), dim=1))
        h2 = F.elu(self.fc2_prior(torch.cat((c, joints, h1), dim=1)))
        s = torch.cat((c, joints, h2), dim=1)
        return self.mu_prior(s), self.logvar_prior(s)

    def forward(self, c):
        mu_prior, logvar_prior = self.encode_prior(c)
        z = mu_prior + logvar_prior
        return z, mu_prior, logvar_prior
    
class MeshDeformerWithJointsDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_joints,
        num_experts,
    ):
        super().__init__()
        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = 1 * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c

        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out
    

class MultiFramePoseWithJointsAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_experts,
        num_joints
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_joints = num_joints

        hidden_size = 256
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_joints
        )
        self.encoder = MeshDeformWithJointsEncoder(*args)
        self.decoder = MeshDeformerWithJointsDecoder(*args, num_experts)

    def encode(self, c, joints):
        _, mu_prior, logvar_prior = self.encoder(c, joints)
        return mu_prior, logvar_prior

    def forward(self, c, joints):
        z, mu_prior, logvar_prior = self.encoder(c, joints)
        return self.decoder(z, c), mu_prior, logvar_prior

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)