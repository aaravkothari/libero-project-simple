import torch
import torch.nn as nn
import torch.nn.functional as F
import os


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ConvEncoder(nn.Module):
    """Nature-CNN trunk over one 128x128 camera.

    One instance per camera rather than a shared trunk: agentview and the wrist
    camera see completely different viewpoints and scales, so shared weights
    would spend capacity straddling both.
    """

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

    # joint_dim=9 is robot0_joint_pos (7) + robot0_gripper_qpos (2), the vector
    # both LiberoObsWrapper._process and DataLoader.get_batch hand back.
    # num_actions=7 is the env's action dim: 6 continuous OSC deltas plus the
    # gripper command, which gets its own head (see forward).
    def __init__(self, input_shape, num_actions, hidden_dim, joint_dim=9,
                 checkpoint_dir='checkpoints', name='bc_network'):

        super(Model, self).__init__()

        self.num_continuous = num_actions - 1

        self.agentview_encoder = ConvEncoder(input_shape[0])
        self.wrist_encoder = ConvEncoder(input_shape[0])

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flat_size = self.agentview_encoder(dummy).shape[1]

        # Both camera embeddings plus proprio are concatenated before linear1.
        self.linear1 = nn.Linear(2 * flat_size + joint_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        # Two heads, because the action vector is two different problems.
        #
        # The 6 OSC deltas are a genuine regression and live in [-1, 1], so
        # tanh + L1 is right for them.
        #
        # The gripper is not: torch.unique over the demo actions returns exactly
        # [-1.0, +1.0], so it is a binary decision. Regressing it alongside the
        # rest made it 58% of the total L1 loss while STILL getting the sign
        # wrong on 13% of training steps -- it crowded out the dims that steer
        # the arm and mispredicted anyway. As a logit trained with BCE it costs
        # a bounded share of the objective and is scored as the classification
        # it actually is.
        self.output = nn.Linear(hidden_dim, self.num_continuous)
        self.gripper = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.apply(weights_init_)

    def forward(self, agentview, wrist, joint_state):
        """Returns (continuous, gripper_logit).

        continuous     (B, 6) in [-1, 1], the OSC position/rotation deltas.
        gripper_logit  (B, 1) raw logit; > 0 selects the +1 gripper command.

        agentview and wrist are (B, 3, 128, 128) float32 in [0, 1].
        joint_state is (B, joint_dim) float32 -- (B, 9) from
        DataLoader.get_batch, and (1, 9) at rollout time after the caller
        unsqueezes the bare (9,) env_wrapper returns.

        NOTE: joint_state is RAW/unnormalized on both paths (see the train/eval
        matching note in env_wrapper.py). Joint angles are order ~1 rad while
        gripper qpos is order ~0.04, and the conv features they sit beside are
        post-ReLU. If training is unstable, normalize -- but then normalize in
        env_wrapper._process with the SAME stats, or you reintroduce the
        train/eval skew.
        """
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
        """Full (B, 7) env action for rollouts.

        The gripper is snapped back to the +/-1 the demos actually contain
        rather than passed through as a soft value -- a half-closed gripper is
        not a command the demonstrations ever issue.
        """
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
