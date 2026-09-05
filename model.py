import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.flatten(1)


class Model(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_dim, joint_dim=9,
                 checkpoint_dir='checkpoints', name='bc_network'):
        super(Model, self).__init__()

        self.num_continuous = num_actions - 1

        self.agentview_encoder = ConvEncoder(input_shape[0])
        self.wrist_encoder = ConvEncoder(input_shape[0])

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_size = self.agentview_encoder(dummy).shape[1]

        self.linear1 = nn.Linear(2 * flat_size + joint_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.output = nn.Linear(hidden_dim, self.num_continuous)
        self.gripper = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.apply(weights_init_)

    def forward(self, agentview, wrist, joint_state):
        x = torch.cat([
            self.agentview_encoder(agentview),
            self.wrist_encoder(wrist),
            joint_state,
        ], dim=1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        return F.tanh(self.output(x)), self.gripper(x)

    @torch.no_grad()
    def act(self, agentview, wrist, joint_state):
        continuous, gripper_logit = self.forward(agentview, wrist, joint_state)

        gripper = torch.where(gripper_logit > 0,
                              torch.ones_like(gripper_logit),
                              -torch.ones_like(gripper_logit))

        return torch.cat([continuous, gripper], dim=1)

    def _path(self, name=None):
        if name is None:
            return self.checkpoint_file
        return os.path.join(self.checkpoint_dir, name)

    def save_checkpoint(self, name=None):
        torch.save(self.state_dict(), self._path(name))

    def load_checkpoint(self, name=None):
        self.load_state_dict(torch.load(self._path(name)))

# ==========================================================================
# ORIGINAL model.py -- image-only CNN, before joint_state was used
# ==========================================================================
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
#
#
# def weights_init_(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight, gain=1)
#         torch.nn.init.constant_(m.bias, 0)
#
#
# class Model(nn.Module):
#
#     def __init__(self, input_shape, num_actions, hidden_dim,
#                  checkpoint_dir='checkpoints', name='bc_network'):
#
#         super(Model, self).__init__()
#
#         self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#
#         with torch.no_grad():
#             dummy = torch.zeros(1, *input_shape)
#             flat_size = self._conv_forward(dummy).shape[1]
#
#         self.linear1 = nn.Linear(flat_size, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, hidden_dim)
#         self.output = nn.Linear(hidden_dim, num_actions)
#
#         self.name = name
#         self.checkpoint_dir = checkpoint_dir
#         self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
#
#         os.makedirs(self.checkpoint_dir, exist_ok=True)
#
#         self.apply(weights_init_)
#
#     def _conv_forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         return x.flatten(1)
#
#     def forward(self, x, joint_state):
#         x = self._conv_forward(x)
#
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         x = F.tanh(self.output(x))
#         return x
#
#     def save_checkpoint(self):
#         torch.save(self.state_dict(), self.checkpoint_file)
#
#     def load_checkpoint(self):
#         self.load_state_dict(torch.load(self.checkpoint_file))
