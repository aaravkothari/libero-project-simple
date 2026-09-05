import numpy as np


class LiberoObsWrapper:
    """Gym-style wrapper over a LIBERO env (OffScreenRenderEnv / ControlEnv).

    Collapses the raw obs dict down to exactly what the policy consumes:
      - agentview (3, H, W) float32 in [0, 1], CHW    <- scene camera
      - wrist     (3, H, W) float32 in [0, 1], CHW    <- eye-in-hand camera
      - joint_state (9,)       float32, RAW (unnormalized) <- joint_pos + gripper_qpos

    Everything else in the obs dict is dropped.

    Not a literal gym.Wrapper subclass: LIBERO's ControlEnv is not a gym.Env
    (no observation_space/action_space), so subclassing gym.Wrapper would trip
    its space assertions. This exposes the same reset()/step() surface instead.

    Train/eval matching (see CONTEXT.md):
      - live keys differ from HDF5: agentview_image / robot0_eye_in_hand_image
        / robot0_joint_pos / robot0_gripper_qpos  (vs agentview_rgb /
        eye_in_hand_rgb / joint_states / gripper_states).
      - image: HWC uint8 [0,255] -> CHW float32 [0,1]. NO vertical flip; the
        stored demos and the live env share the opengl convention.
      - joint_state is raw here because the training loader delivers it raw too. If
        you normalize joint_state for training, normalize it here with the SAME
        per-dim stats or you reintroduce skew.
    """

    def __init__(self, env, image_key="agentview_image",
                 wrist_key="robot0_eye_in_hand_image"):
        self.env = env
        self.image_key = image_key
        self.wrist_key = wrist_key

    def _to_chw(self, image):
        image = np.transpose(image, (2, 0, 1))            # (H,W,3) -> (3, H, W)
        return image.astype(np.float32) / 255.0           # -> [0, 1]

    def _process(self, obs):
        agentview = self._to_chw(obs[self.image_key])
        wrist = self._to_chw(obs[self.wrist_key])

        joint_state = np.concatenate([
            obs["robot0_joint_pos"],                      # (7,) joints
            obs["robot0_gripper_qpos"],                   # (2,) gripper
        ]).astype(np.float32)                             # -> (9,)

        return agentview, wrist, joint_state

    @property
    def horizon(self):
        # Episode length after which robosuite latches its internal done flag
        # (1000 for LIBERO). Stepping past it raises ValueError, and LIBERO's
        # bddl_base_domain.step overwrites the returned done with
        # _check_success(), so the horizon never reaches the caller -- the
        # rollout loop has to bound itself with this.
        return self.env.env.horizon

    def render(self):
        self.env.env.render()

    def render_frame(self, height=512, width=512, camera_name="agentview"):
        """Offscreen RGB frame for video recording: (H, W, 3) uint8, upright.

        Deliberately independent of the observation the policy consumes. The
        offscreen buffer grows on demand (robosuite binding_utils.render), so
        asking for 512x512 here does not disturb the 128x128 obs pipeline --
        the recording stays watchable while the model still sees exactly what
        it was trained on.

        MuJoCo returns frames bottom-up, hence the flip; this is the same
        correction robosuite's own OpenCVRenderer applies before display.
        """
        frame = self.env.env.sim.render(
            camera_name=camera_name, height=height, width=width
        )
        return np.flip(frame, axis=0)

    def reset(self):
        # LIBERO reset() returns a single obs dict (not a gym (obs, info) tuple).
        obs = self.env.reset()
        return self._process(obs)

    def set_init_state(self, init_state):
        # Rollouts start from the demo's fixed init state, not a random reset.
        obs = self.env.set_init_state(init_state)
        return self._process(obs)

    def step(self, action):
        # robosuite returns a 4-tuple (obs, reward, done, info), not gym's 5.
        obs, reward, done, info = self.env.step(action)
        return self._process(obs), reward, done, info

    def seed(self, seed):
        return self.env.seed(seed)

    def close(self):
        return self.env.close()

