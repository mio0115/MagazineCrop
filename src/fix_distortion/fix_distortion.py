import cv2
import numpy as np

from ..utils.misc import reorder_coordinates, resize_with_aspect_ratio


class FixDistortion(object):
    def __init__(self, target_size: tuple[int] = (1024, 1024)):
        self._target_size = target_size

    def perspectiveTransformApproach(self, img, mask):
        amplified_mask = (mask.copy() * 255).astype(np.uint8)

        contours, _ = cv2.findContours(
            amplified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True).reshape(4, 2)
        approx = reorder_coordinates(approx)

        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect).reshape(4, 2)
        box = reorder_coordinates(box)

        mat = cv2.getPerspectiveTransform(
            approx.astype(np.float32), box.astype(np.float32)
        )

        warped = cv2.warpPerspective(img, mat, (int(rect[1][1]), int(rect[1][0])))

        resized_warped, _ = resize_with_aspect_ratio(
            warped, target_size=self._target_size
        )

        return resized_warped

    def __call__(self, img, mask):
        repaired_img = self.perspectiveTransformApproach(img, mask)

        return repaired_img
