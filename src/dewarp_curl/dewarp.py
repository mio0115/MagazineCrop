import numpy as np

from .perspective_transform_approach import PerspectiveTransformApproach
from .thin_plate_spline_approach import ThinPlateSplineApproach


class FixDistortion(object):
    def __init__(self, transform, target_size: tuple[int] = (1024, 1024)):
        self._target_size = target_size
        self._transform = transform

    def __call__(self, img, mask):
        repaired_img = self._transform(img, mask)

        return repaired_img


def compute_edge_len(points: list[np.ndarray]) -> float:
    # compute the length of each edge
    edge_len = []
    for pt1, pt2 in zip(points, points[1:]):
        edge_len.append(np.linalg.norm(pt1 - pt2))

    # return the distance of each vertex from the origin
    # note that the origin is the starting point for each edge (i.e., each corner)
    return np.cumsum(edge_len)
