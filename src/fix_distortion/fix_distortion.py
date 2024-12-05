import cv2
import numpy as np

from ..utils.misc import reorder_coordinates


class CombinationGenerator(object):
    def __init__(self, num_of_items: int):
        self._n = num_of_items

    @property
    def num_of_items(self):
        return self._n

    def __iter__(self):
        self.pt1 = 0
        self.pt2 = 1
        self.pt3 = 2
        self.pt4 = 2

        return self

    def __next__(self):
        self.pt4 += 1
        if self.pt4 == self._n:
            self.pt3 += 1
            self.pt4 = self.pt3 + 1
        if self.pt3 == self._n - 1:
            self.pt2 += 1
            self.pt3 = self.pt2 + 1
            self.pt4 = self.pt3 + 1
        if self.pt2 == self._n - 2:
            self.pt1 += 1
            self.pt2 = self.pt1 + 1
            self.pt3 = self.pt2 + 1
            self.pt4 = self.pt3 + 1

        if self.pt4 == self._n:
            raise StopIteration
        return self.pt1, self.pt2, self.pt3, self.pt4


class FixDistortion(object):
    def __init__(self, args, target_size: tuple[int] = (1024, 1024)):
        self._target_size = target_size
        self._no_resize = args.no_resize

    def perspectiveTransformApproach(self, img, mask):
        padded_img = np.pad(
            img, ((100, 100), (100, 100), (0, 0)), "constant", constant_values=0
        ).astype(np.uint8)
        padded_mask = np.pad(
            mask, ((100, 100), (100, 100)), "constant", constant_values=0
        )
        amplified_mask = (padded_mask * 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            amplified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        polygon = cv2.approxPolyDP(largest_contour, epsilon, True).reshape(-1, 2)
        if polygon.shape[0] < 4:
            print("Polygon is a triangle or a line!!!!")

        hull, _ = FixDistortion.splitPolygon(padded_mask, polygon)
        quad = np.array(
            (
                hull["hull_points"][hull["corners"]["top-left"]],
                hull["hull_points"][hull["corners"]["top-right"]],
                hull["hull_points"][hull["corners"]["bottom-right"]],
                hull["hull_points"][hull["corners"]["bottom-left"]],
            ),
            dtype=np.float32,
        )

        rect = cv2.minAreaRect(hull["hull_points"])
        # align the box with the x-axis AND the y-axis
        # the upper-left corner is the origin
        if rect[-1] < 45:
            box_width, box_height = rect[1]
        else:
            box_height, box_width = rect[1]
        aligned_box = np.array(
            [
                [0, 0],
                [box_width, 0],
                [box_width, box_height],
                [0, box_height],
            ],
            dtype=np.float32,
        )

        mat = cv2.getPerspectiveTransform(quad, aligned_box)
        dewarped = cv2.warpPerspective(
            padded_img, mat, (int(box_width), int(box_height))
        )

        return dewarped

    @staticmethod
    def splitPolygon(mask: np.ndarray, polygon: np.ndarray):
        if polygon.shape[0] == 4:
            quad = reorder_coordinates(polygon)
            return {
                "hull_points": quad,
                "corners": {
                    "top-left": 0,
                    "top-right": 1,
                    "bottom-right": 2,
                    "bottom-left": 3,
                },
            }, None

        num_pts = polygon.shape[0]
        # find the 4 points that form the rectangle with the largest area
        curr_choice = {"pts": None, "area": 0}
        for pt1, pt2, pt3, pt4 in CombinationGenerator(num_pts):
            rect = np.array(
                [polygon[pt1], polygon[pt2], polygon[pt3], polygon[pt4]], np.int32
            )
            rect = reorder_coordinates(rect)

            # compute the number of points inside the rectangle
            mask_rect = np.zeros_like(mask)
            cv2.fillConvexPoly(mask_rect, rect, 1)
            area_mask_rect = np.sum(mask * mask_rect)
            # update the current choice
            if area_mask_rect > curr_choice["area"]:
                curr_choice["pts"] = (pt1, pt2, pt3, pt4)
                curr_choice["area"] = area_mask_rect
        # then, we sort the points to determine the 4 corners of the rectangle
        dist = np.take(np.sum(np.abs(polygon), axis=1), curr_choice["pts"], 0)
        diff = np.take(np.diff(np.abs(polygon), axis=1), curr_choice["pts"], 0)
        # the origin is at the top-left corner
        # the y-axis is flipped
        corner_indices = {
            "top-left": curr_choice["pts"][np.argmin(dist)],
            "top-right": curr_choice["pts"][np.argmin(diff)],
            "bottom-left": curr_choice["pts"][np.argmax(diff)],
            "bottom-right": curr_choice["pts"][np.argmax(dist)],
        }

        return {
            "hull_points": polygon,
            "corners": corner_indices,
        }, None

    @staticmethod
    def splitBoxes(convex_hull, quads: list[np.ndarray], box: np.ndarray):
        edges_len = {"top": None, "right": None, "bottom": None, "left": None}
        edge_vertices = {"top": [], "right": [], "bottom": [], "left": []}

        # find the belonging of each vertex to the edges
        # note that the corners belong to 2 edges
        curr_edge = "top"
        for idx, pt in enumerate(convex_hull["hull_points"].tolist()):
            if idx in convex_hull["corners"].values():
                edge_vertices[curr_edge].append(pt)
                if idx == convex_hull["corners"]["top-right"]:
                    curr_edge = "right"
                elif idx == convex_hull["corners"]["bottom-right"]:
                    curr_edge = "bottom"
                elif idx == convex_hull["corners"]["bottom-left"]:
                    curr_edge = "left"
            edge_vertices[curr_edge].append(pt)
        edge_vertices["left"].append(edge_vertices["top"][0])

        # compute the length ratio of each edge
        # length ratio is the
        #   the distance of each vertex from the corner
        # ------------------------------------------------
        #        the distance from corner to corner
        for edge in edge_vertices.keys():
            edges_len[edge] = np.array(
                compute_edge_len(edge_vertices[edge]), np.float32
            )
            edges_len[edge] = edges_len[edge] / edges_len[edge][-1]

        return

    def __call__(self, img, mask):
        repaired_img = self.perspectiveTransformApproach(img, mask)

        return repaired_img


def make_vertices_even(hull: np.ndarray) -> np.ndarray:
    n_vertices = hull.shape[0]
    if n_vertices % 2 == 0:
        return hull

    # add a new point in the middle of the longest edge
    max_dist = 0
    max_idx = -1
    new_vertex = None
    for i in range(n_vertices):
        dist = np.linalg.norm(hull[i] - hull[(i + 1) % n_vertices])
        if dist > max_dist:
            max_dist = dist
            max_idx = i
            new_vertex = (hull[i] + hull[(i + 1) % n_vertices]) / 2

    # insert the new vertex into the hull
    new_hull = np.insert(hull, max_idx + 1, new_vertex, 0)
    return new_hull


def compute_edge_len(points: list[np.ndarray]) -> float:
    # compute the length of each edge
    edge_len = []
    for pt1, pt2 in zip(points, points[1:]):
        edge_len.append(np.linalg.norm(pt1 - pt2))

    # return the distance of each vertex from the origin
    # note that the origin is the starting point for each edge (i.e., each corner)
    return np.cumsum(edge_len)


if __name__ == "__main__":
    for i in CombinationGenerator(6):
        print(i)
