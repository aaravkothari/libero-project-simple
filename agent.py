import contextlib
import io
import os
from collections import deque
import torch

from model import Model

import cv2
import numpy as np
from dataloader import DataLoader
from env_wrapper import LiberoObsWrapper

import torch.nn.functional as F

with contextlib.redirect_stderr(io.StringIO()):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.env_wrapper import ControlEnv

class Agent:
    def __init__(self, eval=False, lr=0.0001):

        dataset_filename = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"

        # hdf5_cache_mode="all": images were otherwise re-read from disk on every
        # single sample -- 43.4 ms/batch vs 2.5 ms cached, and the whole cache is
        # ~4.6 GB RSS at seq_len=1. Dominates training time at 100k iterations.
        self.dl = DataLoader(dataset_filename=dataset_filename,
                             hdf5_cache_mode="all")

        task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = task_suite.get_task(0)
        task_bddl = os.path.join(get_libero_path("bddl_files"),
                                task.problem_folder, task.bddl_file)

        self.eval = eval

        if self.eval:
            # has_renderer=False: the on-screen path builds robosuite's
            # OpenCVRenderer, whose render() calls cv2.imshow -- that needs an X
            # display and SIGABRTs on a headless box. Video is recorded off the
            # offscreen renderer instead (see test()).
            self.env = ControlEnv(bddl_file_name=task_bddl,
                                has_renderer=False,
                                has_offscreen_renderer=True,
                                use_camera_obs=True,
                                camera_heights=128,
                                camera_widths=128)
        else:
            self.env = OffScreenRenderEnv(bddl_file_name=task_bddl,
                                        camera_heights=128,
                                        camera_widths=128)

        self.env.seed(0)

        self.env = LiberoObsWrapper(self.env)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.model = Model(input_shape=(3,128,128), num_actions=7, hidden_dim=256).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        agentview, wrist, joint_state = self.env.reset()

    def train(self, epochs, batch_size, gripper_weight=1.0):
        
        lowest_loss = 100

        # Checkpoint on the mean of the last 100 iterations, not on a single
        # random batch of 32. Over 1000 saves, the single-batch minimum is
        # mostly a measure of which batch happened to be easy.
        recent = deque(maxlen=100)

        for i in range(epochs):

            batch = self.dl.get_batch(batch_size=batch_size)

            agentview = batch['agentview'].to(self.device)
            wrist = batch['wrist'].to(self.device)
            joint_states = batch['joint_state'].to(self.device)
            actions = batch['actions'].to(self.device)

            continuous_pred, gripper_logit = self.model(agentview, wrist, joint_states)

            # Continuous OSC deltas: regression. Gripper: binary, and the demo
            # commands are exactly -1/+1, so map to the 0/1 BCE wants.
            continuous_loss = F.l1_loss(continuous_pred, actions[:, :-1])
            gripper_loss = F.binary_cross_entropy_with_logits(
                gripper_logit, (actions[:, -1:] > 0).float())

            loss = continuous_loss + gripper_weight * gripper_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            recent.append(loss.item())

            if i % 100 == 0:
                smoothed = sum(recent) / len(recent)

                print(f"Episode: {i}, Loss: {smoothed:.4f} "
                      f"(cont {continuous_loss.item():.4f}, "
                      f"grip {gripper_loss.item():.4f})")

                if smoothed < lowest_loss:
                    lowest_loss = smoothed
                    self.model.save_checkpoint()
                    print(f"\nSaved checkpoint at episode: {i}\n")

        # Also keep the final weights. "Best training loss" and "last" are not
        # the same policy, and neither is reliably the better rollout -- with no
        # validation split there is nothing to arbitrate, so keep both and test
        # them.
        self.model.save_checkpoint(name="bc_network_final")
        print(f"Saved final weights (best smoothed loss was {lowest_loss:.4f})")

    def test(self, video_path="rollout.mp4", steps=None, fps=30, video_size=512,
             checkpoint_name=None):
        self.model.load_checkpoint(name=checkpoint_name)
        self.model.eval()

        # Never step past the horizon -- see LiberoObsWrapper.horizon.
        steps = self.env.horizon if steps is None else min(steps, self.env.horizon)

        agentview, wrist, joint_state = self.env.reset()

        # cv2.VideoWriter needs no display -- only cv2.imshow does. mp4v is the
        # codec the stock opencv-python wheel ships with; pair it with .mp4.
        writer = cv2.VideoWriter(video_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps,
                                 (video_size, video_size))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for {video_path}")

        success = False
        try:
            for i in range(steps):
                av_t = torch.as_tensor(agentview, device=self.device).unsqueeze(0)
                wr_t = torch.as_tensor(wrist, device=self.device).unsqueeze(0)
                js_t = torch.as_tensor(joint_state, device=self.device).unsqueeze(0)

                # act() is @torch.no_grad and snaps the gripper back to +/-1.
                action = self.model.act(av_t, wr_t, js_t).squeeze(0).cpu().numpy()

                (agentview, wrist, joint_state), reward, done, info = self.env.step(action)

                # Rendered separately from the policy's 128x128 observation, so
                # the video is watchable without changing the model's input.
                # ascontiguousarray because the RGB->BGR slice leaves negative
                # strides, which VideoWriter will not accept.
                frame = self.env.render_frame(height=video_size, width=video_size)
                writer.write(np.ascontiguousarray(frame[:, :, ::-1]))

                if done:
                    success = True
                    print(f"Success at step {i}")
                    break
        finally:
            writer.release()

        if not success:
            print(f"No success within {steps} steps")
        print(f"Wrote {video_path}")

    def close(self):
        self.env.close()
