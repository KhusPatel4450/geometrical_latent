from typing import NamedTuple, Optional
from dataclasses import dataclass
import random
import math

import torch

from .single import DotDataset, DotDatasetConfig
from .wall_utils import *


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor
    bias_angle: torch.Tensor
    wall_x: torch.Tensor
    door_y: torch.Tensor


@dataclass
class WallDatasetConfig(DotDatasetConfig):
    fix_wall: bool = True
    fix_wall_batch_k: Optional[int] = None
    wall_padding: int = 20
    door_padding: int = 10
    wall_width: int = 1
    door_space: int = 2
    exclude_wall_train: str = ""
    exclude_door_train: str = ""
    only_wall_val: str = ""
    only_door_val: str = ""
    fix_wall_location: Optional[int] = None
    fix_door_location: Optional[int] = None
    num_train_layouts: Optional[int] = -1


class WallDataset(DotDataset):
    def __init__(self, config: WallDatasetConfig, *args, **kwargs):
        layouts, other_layouts = generate_wall_layouts(config)
        self.layouts = layouts
        super().__init__(config, *args, **kwargs)

    def render_location(self, locations):
        states = super().render_location(locations)
        return states

    def generate_cross_wall_points(self, wall_locs, action_padding=0):
        bs = wall_locs.size(0)
        left_wall_locs = wall_locs - self.config.wall_width // 2
        right_wall_locs = wall_locs + self.config.wall_width // 2

        min_val = self.config.border_wall_loc - 1 + 0.01
        max_val = self.config.img_size - self.config.border_wall_loc - 0.01

        left_x = sample_uniformly_between(
            torch.full((bs,), min_val).to(wall_locs.device),
            left_wall_locs - action_padding,
        )
        right_x = sample_uniformly_between(
            right_wall_locs + action_padding,
            torch.full((bs,), max_val).to(wall_locs.device),
        )
        left_y = sample_uniformly_between(
            torch.full((bs,), min_val).to(wall_locs.device),
            torch.full((bs,), max_val).to(wall_locs.device),
        )
        right_y = sample_uniformly_between(
            torch.full((bs,), min_val).to(wall_locs.device),
            torch.full((bs,), max_val).to(wall_locs.device),
        )

        left_pos = torch.stack([left_x, left_y]).transpose(0, 1)
        right_pos = torch.stack([right_x, right_y]).transpose(0, 1)
        return left_pos, right_pos

    def generate_cross_wall_state_and_actions(self, wall_locs=None, door_locs=None, n_steps=17):
        bs = door_locs.size(0)
        left_wall_locs = wall_locs - self.config.wall_width // 2
        right_wall_locs = wall_locs + self.config.wall_width // 2

        x = sample_uniformly_between(left_wall_locs, right_wall_locs)
        y = sample_truncated_norm(
            upper_bound=door_locs + self.config.door_space,
            lower_bound=door_locs - self.config.door_space,
            mean=door_locs,
        ).to(door_locs.device)

        loc_at_door = torch.stack([x, y]).transpose(0, 1)
        step_idxs = torch.randint(1, n_steps, size=x.shape)

        angles = torch.empty(bs)
        for i in range(bs):
            angles[i] = torch.pi + (torch.rand(1) - 0.5) * torch.pi / 2

        angles = self.angle_to_vec(angles).to(self.device)
        actions_dir_left, _ = self.generate_actions(n_steps, bias_angle=angles)
        actions_dir_right, _ = self.generate_actions(n_steps, bias_angle=-1 * angles)

        cw_actions = torch.zeros((bs, n_steps - 1, 2))
        cw_start_loc = torch.zeros((bs, 2))

        for i in range(bs):
            step = step_idxs[i]
            traj = torch.cat([
                torch.flip(actions_dir_left[i][:step], dims=[0]) * -1,
                actions_dir_right[i][1: n_steps - step],
            ])
            step_sum_before_door = traj[:step].sum(dim=0)

            if random.random() < 0.5:
                traj = torch.flip(traj, dims=[0]) * -1
                step_sum_before_door = traj[: n_steps - step].sum(dim=0)

            cw_actions[i] = traj
            cw_start_loc[i] = loc_at_door[i] - step_sum_before_door

        min_val = self.config.border_wall_loc - 1 + 0.01
        max_val = self.config.img_size - self.config.border_wall_loc - 0.01
        cw_start_loc = torch.clamp(cw_start_loc, min=min_val, max=max_val)

        return cw_start_loc, cw_actions, torch.zeros_like(cw_actions)

    def check_wall_intersection(self, current_location, next_location, walls):
        half_width = self.config.wall_width // 2
        wall_left = walls - half_width
        wall_right = walls + half_width

        current_right = current_location[:, 0] <= wall_right
        next_right = next_location[:, 0] <= wall_right
        current_left = current_location[:, 0] >= wall_left
        next_left = next_location[:, 0] >= wall_left

        inside_wall = (current_right & current_left) != (next_right & next_left)
        across_wall = (current_right != next_right) & (current_left != next_left)

        return inside_wall | across_wall

    def check_pass_through_door(self, current_location, next_location, wall_loc, door_loc):
        half_width = self.config.wall_width // 2
        left_wall = wall_loc - half_width
        right_wall = wall_loc + half_width

        d = next_location - current_location
        a = d[1] / d[0]
        b = current_location[1] - a * current_location[0]

        if (
            torch.sign(left_wall - current_location[0])
            * torch.sign(left_wall - next_location[0])
            < 0
        ):
            y_left = a * left_wall + b
            pass_left_wall = (
                door_loc - self.config.door_space <= y_left <= door_loc + self.config.door_space
            )
        else:
            pass_left_wall = True

        if (
            torch.sign(right_wall - current_location[0])
            * torch.sign(right_wall - next_location[0])
            < 0
        ):
            y_right = a * right_wall + b
            pass_right_wall = (
                door_loc - self.config.door_space <= y_right <= door_loc + self.config.door_space
            )
        else:
            pass_right_wall = True

        return pass_left_wall and pass_right_wall

    @staticmethod
    def segments_intersect(A, B):
        A0, A1 = A[:, 0], A[:, 1]
        B0, B1 = B[:, 0], B[:, 1]
        dA = A1 - A0
        dB = B1 - B0

        def cross_2d(v, w):
            return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]

        B0_to_A0 = B0 - A0
        B1_to_A0 = B1 - A0
        A0_to_B0 = A0 - B0
        A1_to_B0 = A1 - B0

        cross_A_B0 = cross_2d(dA, B0_to_A0)
        cross_A_B1 = cross_2d(dA, B1_to_A0)
        cross_B_A0 = cross_2d(dB, A0_to_B0)
        cross_B_A1 = cross_2d(dB, A1_to_B0)

        intersect_A = cross_A_B0 * cross_A_B1 < 0
        intersect_B = cross_B_A0 * cross_B_A1 < 0

        return (intersect_A & intersect_B).long()

    def check_wall_width_intersection(self, locations, next_locations, walls, doors):
        disp = torch.stack([locations, next_locations], dim=1)
        deltas = next_locations - locations
        upwards = deltas[:, 1] > 0
        downwards = deltas[:, 1] < 0

        left_wall = walls - self.config.wall_width // 2
        right_wall = walls + self.config.wall_width // 2
        door_bot = doors - self.config.door_space
        door_top = doors + self.config.door_space

        top_left = torch.stack([left_wall, door_top], dim=1)
        top_right = torch.stack([right_wall, door_top], dim=1)
        bot_left = torch.stack([left_wall, door_bot], dim=1)
        bot_right = torch.stack([right_wall, door_bot], dim=1)

        top_seg = torch.stack([top_left, top_right], dim=1)
        bot_seg = torch.stack([bot_left, bot_right], dim=1)

        top_intersect = self.segments_intersect(disp, top_seg)
        bot_intersect = self.segments_intersect(disp, bot_seg)

        return (top_intersect & upwards) | (bot_intersect & downwards)

    def generate_transitions(self, location, actions, bias_angle, walls):
        locations = [location]
        for i in range(actions.shape[1]):
            next_location = self.generate_transition(locations[-1], actions[:, i])
            left_border = torch.zeros_like(walls[0])
            left_border[:] = self.config.border_wall_loc - 1
            right_border = torch.zeros_like(walls[0])
            right_border[:] = self.config.img_size - self.config.border_wall_loc
            top_border, bot_border = left_border, right_border

            check_border_intersection = (
                ((torch.sign(locations[-1][:, 0] - left_border) * torch.sign(next_location[:, 0] - left_border)) <= 0)
                | ((torch.sign(locations[-1][:, 0] - right_border) * torch.sign(next_location[:, 0] - right_border)) <= 0)
                | ((torch.sign(locations[-1][:, 1] - top_border) * torch.sign(next_location[:, 1] - top_border)) <= 0)
                | ((torch.sign(locations[-1][:, 1] - bot_border) * torch.sign(next_location[:, 1] - bot_border)) <= 0)
            )

            check_wall_intersection = self.check_wall_intersection(locations[-1], next_location, walls[0])
            check_wall_width_intersection = self.check_wall_width_intersection(
                locations=locations[-1], next_locations=next_location, walls=walls[0], doors=walls[1],
            )

            check_intersection = check_border_intersection | check_wall_intersection | check_wall_width_intersection

            for j in check_intersection.nonzero():
                if check_border_intersection[j] or check_wall_width_intersection[j]:
                    next_location[j] = locations[-1][j].clone()
                else:
                    if not self.check_pass_through_door(
                        current_location=locations[-1][j][0],
                        next_location=next_location[j][0],
                        wall_loc=walls[0][j],
                        door_loc=walls[1][j],
                    ):
                        next_location[j] = locations[-1][j].clone()

            locations.append(next_location)

        locations = torch.stack(locations, dim=1).unsqueeze(dim=-2)
        actions = actions.unsqueeze(dim=-2)
        states = self.render_location(locations)
        walls_rendered = self.render_walls(*walls).unsqueeze(1).unsqueeze(1)
        walls_rendered = walls_rendered.repeat(1, states.shape[1], 1, 1, 1)
        states_with_walls = torch.cat([states, walls_rendered], dim=-3)

        if self.config.n_steps_reduce_factor > 1:
            states_with_walls = states_with_walls[:, :: self.config.n_steps_reduce_factor]
            locations = locations[:, :: self.config.n_steps_reduce_factor]
            reduced_chunks = actions.shape[1] // self.config.n_steps_reduce_factor
            action_chunks = torch.chunk(actions, chunks=reduced_chunks, dim=1)
            actions = torch.cat(
                [torch.sum(chunk, dim=1, keepdim=True) for chunk in action_chunks], dim=1
            )

        return WallSample(
            states=states_with_walls,
            locations=locations,
            actions=actions,
            bias_angle=bias_angle,
            wall_x=None,
            door_y=None,
        )

    def sample_walls(self):
        layout_codes = list(self.layouts.keys())
        if self.config.fix_wall_batch_k is not None:
            layout_codes = random.sample(layout_codes, self.config.fix_wall_batch_k)

        weights = [1] * len(layout_codes)
        sampled_codes = random.choices(layout_codes, weights=weights, k=self.config.batch_size)
        wall_locs, door_locs, types = [], [], []

        for code in sampled_codes:
            attr = self.layouts[code]
            wall_locs.append(attr["wall_pos"])
            door_locs.append(attr["door_pos"])
            types.append(attr["type"])

        wall_locs = torch.tensor(wall_locs, device=self.device)
        door_locs = torch.tensor(door_locs, device=self.device)
        return (wall_locs, door_locs)

    def render_walls(self, wall_locs, hole_locs):
        x = torch.arange(0, self.config.img_size, device=self.device)
        y = torch.arange(0, self.config.img_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        grid_x = grid_x.unsqueeze(0).repeat(self.config.batch_size, 1, 1)
        grid_y = grid_y.unsqueeze(0).repeat(self.config.batch_size, 1, 1)

        wall_locs_r = wall_locs.view(self.config.batch_size, 1, 1).repeat(1, self.config.img_size, self.config.img_size)
        hole_locs_r = hole_locs.view(self.config.batch_size, 1, 1).repeat(1, self.config.img_size, self.config.img_size)

        offset = self.config.wall_width // 2
        wall_mask = (wall_locs_r - offset <= grid_x) & (grid_x <= wall_locs_r + offset)

        res = (
            wall_mask * (
                (hole_locs_r < grid_y - self.config.door_space)
                + (hole_locs_r > grid_y + self.config.door_space)
            )
        ).float()

        border_wall_loc = self.config.border_wall_loc
        res[:, :, border_wall_loc - 1] = 1
        res[:, :, -border_wall_loc] = 1
        res[:, border_wall_loc - 1, :] = 1
        res[:, -border_wall_loc, :] = 1

        return res
