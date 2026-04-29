import copy
import random
import torch
from scipy.stats import truncnorm
import numpy as np

def normalize_images(img):
    if img.dim() == 2:
        max_val = torch.max(img)
        return img / max_val
    elif img.dim() == 5:
        bs, T, _, H, W = img.shape
        normalized_img = torch.zeros_like(img)
        for b in range(bs):
            for t in range(T):
                single_img = img[b, t, 0]
                max_val = torch.max(single_img)
                normalized_img[b, t, 0] = single_img / max_val
        return normalized_img
    else:
        raise ValueError("Unsupported input shape. Expected either (h, w) or (bs, T, 1, h, w).")

def generate_wall_layouts(wall_config):
    """
    Generate possible layouts for train and validation
    """
    img_size = wall_config.img_size
    wall_padding = wall_config.wall_padding
    door_padding = wall_config.door_padding

    fix_wall = wall_config.fix_wall
    fix_door_location = wall_config.fix_door_location
    fix_wall_location = wall_config.fix_wall_location
    num_train_layouts = wall_config.num_train_layouts

    def extract_min_max(code):
        vals = [int(x) for x in code.split("-") if x]
        if len(vals) == 2:
            min_val, max_val = vals[0], vals[1]
        elif len(vals) == 1:
            min_val, max_val = vals[0], vals[0]
        else:
            return []
        return list(range(min_val, max_val + 1))

    exclude_wall_train = extract_min_max(wall_config.exclude_wall_train)
    exclude_door_train = extract_min_max(wall_config.exclude_door_train)
    only_wall_val = extract_min_max(wall_config.only_wall_val)
    only_door_val = extract_min_max(wall_config.only_door_val)

    assert all(
        len(arr) == 0
        for arr in [exclude_wall_train, exclude_door_train, only_wall_val, only_door_val]
    ) or all(
        len(arr) > 0
        for arr in [exclude_wall_train, exclude_door_train, only_wall_val, only_door_val]
    )

    if fix_wall:
        layouts = {
            f"v_wall{fix_wall_location}_door{fix_door_location}": {
                "type": "v",
                "wall_pos": fix_wall_location,
                "door_pos": fix_door_location,
            }
        }
        return layouts, None

    wall_x_values = list(range(wall_padding, img_size - wall_padding))
    door_y_values = list(range(door_padding, img_size - door_padding))

    layouts = {}
    other_layouts = {}

    for wall_pos in wall_x_values:
        for door_pos in door_y_values:
            code = f"wall{wall_pos}_door{door_pos}"

            to_exclude_for_train = (
                wall_pos in exclude_wall_train and door_pos in exclude_door_train
            )
            to_include_for_val = wall_pos in only_wall_val and door_pos in only_door_val

            if not to_exclude_for_train:
                layouts[f"v_{code}"] = {"type": "v", "wall_pos": wall_pos, "door_pos": door_pos}
                layouts[f"h_{code}"] = {"type": "h", "wall_pos": wall_pos, "door_pos": door_pos}

            if to_include_for_val:
                other_layouts[f"v_{code}"] = {"type": "v", "wall_pos": wall_pos, "door_pos": door_pos}
                other_layouts[f"h_{code}"] = {"type": "h", "wall_pos": wall_pos, "door_pos": door_pos}

    if not wall_config.train and exclude_wall_train:
        return other_layouts, None
    else:
        return layouts, None


def sample_uniformly_between(a, b):
    random_values = torch.FloatTensor(a.size()).uniform_(0, 1).to(a.device)
    c = a + (b - a) * random_values
    return c


def sample_truncated_norm(upper_bound, lower_bound, mean, std=1.4):
    upper_bound_np = upper_bound.cpu().float().numpy()
    lower_bound_np = lower_bound.cpu().float().numpy()
    mean_np = mean.cpu().float().numpy()

    samples = np.zeros_like(mean_np)

    for i in range(len(mean_np)):
        a, b = (lower_bound_np[i] - mean_np[i]) / std, (upper_bound_np[i] - mean_np[i]) / std
        samples[i] = truncnorm.rvs(a, b, loc=mean_np[i], scale=std)

    return torch.from_numpy(samples)
