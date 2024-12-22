import math
from typing import Callable

import numpy as np
import cv2
from scipy.interpolate import PchipInterpolator, Rbf

from ..utils.paper_size import aspect_ratio
from ..utils.misc import compute_dist_2d


def sort_points_clockwise(pts: np.ndarray):
    center = np.mean(pts, axis=0)

    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]

    return sorted_pts


def compute_intersection(pt: tuple[float], segments: np.ndarray):
    sort_segs = segments[segments[..., 0].argsort()]
    idx = np.searchsorted(sort_segs[..., 0], pt[0], side="left")
    pt_0, pt_1 = sort_segs[idx - 1], sort_segs[idx]

    ratio = abs(pt[0] - pt_0[0]) / abs(pt_1[0] - pt_0[0])
    intersection = pt_0 + ratio * (pt_1 - pt_0)

    return intersection


def make_parallel(pts, corners):
    left_avg_x = (pts[corners["top-left"], 0] + pts[corners["bottom-left"], 0]) / 2
    right_avg_x = (pts[corners["top-right"], 0] + pts[corners["bottom-right"], 0]) / 2

    pts[corners["top-left"]][0] = left_avg_x
    pts[corners["bottom-left"]][0] = left_avg_x
    pts[corners["top-right"]][0] = right_avg_x
    pts[corners["bottom-right"]][0] = right_avg_x

    return pts


def compute_weights(xs, spline, weight_fn: Callable = lambda x: x):
    # compute the slope based on midpoints of xs
    mid_xs = (xs[:-1] + xs[1:]) / 2
    dy_dx = spline.derivative(nu=1)(mid_xs)
    # compute the magnitude of the slope
    slope_magnitude = np.abs(dy_dx)
    # compute weights based on slope magnitude
    raw_weights = np.vectorize(weight_fn)(slope_magnitude.copy())

    # if the total weight is close to zero, then set all weights to 1
    if np.isclose(np.sum(raw_weights), 0):
        raw_weights = np.ones_like(weights)
    # normalize the weights
    weights = raw_weights / raw_weights.sum()
    return weights


def get_grid(top_edge, bottom_edge, interval: int = 20):
    target_height = 0
    for src_x, end_x in zip(
        np.arange(top_edge["start"][0], top_edge["end"][0], dtype=np.float32, step=1),
        np.arange(
            bottom_edge["start"][0], bottom_edge["end"][0], dtype=np.float32, step=1
        ),
    ):
        src_y = top_edge["interpolator"](src_x)
        end_y = bottom_edge["interpolator"](end_x)
        dist = compute_dist_2d((src_x, src_y), (end_x, end_y))
        target_height = max(target_height, dist)

    # suppose that the aspect ratio is sqrt(2)
    target_width = target_height / aspect_ratio

    sample_top_xs = np.linspace(top_edge["start"][0], top_edge["end"][0], interval)
    sample_top_ys = top_edge["interpolator"](sample_top_xs)
    sample_bottom_xs = np.linspace(
        bottom_edge["start"][0], bottom_edge["end"][0], interval
    )
    sample_bottom_ys = bottom_edge["interpolator"](sample_bottom_xs)

    grid = [np.linspace(top_edge["start"], bottom_edge["start"], interval)]
    for i in range(1, interval - 1):
        grid.append(
            np.linspace(
                (sample_top_xs[i], sample_top_ys[i]),
                (sample_bottom_xs[i], sample_bottom_ys[i]),
                interval,
            )
        )
    grid.append(np.linspace(top_edge["end"], bottom_edge["end"], interval))
    source_grid = np.array(grid, dtype=np.float32)

    weights = compute_weights(
        xs=sample_top_xs,
        spline=top_edge["interpolator"],
        weight_fn=lambda x: x,
    )

    original_segments = []
    for i in range(interval - 1):
        dx = sample_top_xs[i + 1] - sample_top_xs[i]
        dy = sample_top_ys[i + 1] - sample_top_ys[i]
        segment_length = math.sqrt(dx * dx + dy * dy)
        original_segments.append(segment_length)
    original_segments = np.array(original_segments, dtype=np.float32)

    projected_width = np.sum(original_segments)
    delta_lengths = weights * max(target_width - projected_width, 0)

    adjusted_segments = original_segments + delta_lengths

    grid = [np.linspace((0, 0), (0, target_height), interval)]
    curr_x = 0
    for seg_len in adjusted_segments:
        curr_x += seg_len
        grid.append(np.linspace((curr_x, 0), (curr_x, target_height), interval))
    target_grid = np.array(grid, dtype=np.float32)

    return source_grid, target_grid


def get_meshgrid(pts, corners, interval: int = 20):
    # find the corresponding points for the top edge
    # also find the corresponding points for the bottom edge
    corresponding_pts = {
        tuple(pts[corners["top-left"]]): tuple(pts[corners["bottom-left"]]),
        tuple(pts[corners["top-right"]]): tuple(pts[corners["bottom-right"]]),
    }
    for idx in range(corners["top-left"] + 1, corners["top-right"]):
        corresponding_pts[tuple(pts[idx])] = tuple(
            compute_intersection(
                pt=pts[idx],
                segments=pts[corners["bottom-right"] : corners["bottom-left"] + 1, :],
            )
        )

    for idx in range(corners["bottom-right"] + 1, corners["bottom-left"]):
        top_pt = compute_intersection(
            pt=pts[idx], segments=pts[corners["top-left"] : corners["top-right"] + 1, :]
        )
        corresponding_pts[tuple(top_pt)] = tuple(pts[idx])

    top_edge_pts = np.array(
        sorted(list(corresponding_pts.keys()), key=lambda x: x[0]), dtype=np.float32
    )
    bot_edge_pts = np.array(
        sorted(list(corresponding_pts.values()), key=lambda x: x[0]), dtype=np.float32
    )

    top_interpolator = PchipInterpolator(x=top_edge_pts[:, 0], y=top_edge_pts[:, 1])
    bot_interpolator = PchipInterpolator(x=bot_edge_pts[:, 0], y=bot_edge_pts[:, 1])

    top_edge = {
        "interpolator": top_interpolator,
        "start": top_edge_pts[0],
        "end": top_edge_pts[-1],
    }
    bot_edge = {
        "interpolator": bot_interpolator,
        "start": bot_edge_pts[0],
        "end": bot_edge_pts[-1],
    }
    src_grid, tgt_grid = get_grid(top_edge, bot_edge)

    return src_grid, tgt_grid


def crop(image: np.ndarray, src_grid: np.ndarray):
    src_x = src_grid[:, :, 0].flatten()
    src_y = src_grid[:, :, 1].flatten()

    min_x, max_x = int(np.min(src_x)), int(np.max(src_x))
    min_y, max_y = int(np.min(src_y)), int(np.max(src_y))

    return (
        image[min_y : max_y + 1, min_x : max_x + 1, :],
        src_grid - np.array([min_x, min_y]),
    )


def apply_tps(src_grid: np.ndarray, tgt_grid: np.ndarray, image: np.ndarray):
    src_x = src_grid[:, :, 0].flatten()
    src_y = src_grid[:, :, 1].flatten()

    tgt_x = tgt_grid[:, :, 0].flatten()
    tgt_y = tgt_grid[:, :, 1].flatten()

    # note that we build mapping from target to source
    rbf_x = Rbf(tgt_x, tgt_y, src_x, function="thin_plate")
    rbf_y = Rbf(tgt_x, tgt_y, src_y, function="thin_plate")

    grid_x, grid_y = np.meshgrid(
        np.arange(tgt_grid[-1, -1, 0]), np.arange(tgt_grid[-1, -1, 1])
    )
    grid_x_flat, grid_y_flat = grid_x.flatten(), grid_y.flatten()

    map_x = rbf_x(grid_x_flat, grid_y_flat).reshape(grid_y.shape).astype(np.float32)
    map_y = rbf_y(grid_x_flat, grid_y_flat).reshape(grid_y.shape).astype(np.float32)

    warped_image = cv2.remap(
        image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

    return warped_image


if __name__ == "__main__":
    import os
    import json
    from PIL import Image

    with open(
        os.path.join("/", "home", "daniel", "Downloads", "no6-1022_162130_rot_1.json"),
        mode="r",
    ) as fin:
        labels = json.load(fin)

    pil_image = Image.open(
        os.path.join("/", "home", "daniel", "Downloads", "no6-1022_162130_rot_1.jpg")
    )
    edge_scale = 3
    im = np.array(pil_image, dtype=np.uint8)[..., ::-1]
    h, w, _ = im.shape
    scaled_im = cv2.resize(im, (w // edge_scale, h // edge_scale))

    top_left_ref = (
        np.array(labels["shapes"][1]["points"][0], dtype=np.float32) / edge_scale
    )

    pts = np.array(labels["shapes"][0]["points"], dtype=np.float32) / edge_scale
    # sort the points starting from the top-left corner in the clockwise direction
    sorted_pts = sort_points_clockwise(pts)

    min_dist = 100000
    min_idx = 0
    for i, pt in enumerate(sorted_pts):
        dist = np.linalg.norm(top_left_ref - pt)
        if dist < min_dist:
            min_dist = dist
            min_idx = i

    sorted_pts = np.roll(sorted_pts, -min_idx, axis=0)

    corners = {
        "top-left": np.argmin(sorted_pts[:, 0] + sorted_pts[:, 1]),
        "top-right": np.argmax(sorted_pts[:, 0] - sorted_pts[:, 1]),
        "bottom-right": np.argmax(sorted_pts[:, 0] + sorted_pts[:, 1]),
        "bottom-left": np.argmin(sorted_pts[:, 0] - sorted_pts[:, 1]),
    }

    sorted_pts = make_parallel(sorted_pts, corners)

    src_grid, tgt_grid = get_meshgrid(sorted_pts, corners)

    cropped_im, cropped_src_grid = crop(scaled_im, src_grid)

    warped_im = apply_tps(
        src_grid=cropped_src_grid, tgt_grid=tgt_grid, image=cropped_im
    )

    cv2.namedWindow("warped", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("warped", 1024, 1024)
    cv2.imshow("warped", warped_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
