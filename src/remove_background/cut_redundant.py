from enum import Enum

import numpy as np
import cv2


class ImageType(Enum):
    TALL = 0
    WIDE = 1


def crop_image(image: np.ndarray, is_gray: bool = False):
    if not is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_type = ImageType.TALL if image.shape[0] > image.shape[1] else ImageType.WIDE

    image = cv2.equalizeHist(image)

    height_neglect, width_neglect = 0.02, 0.02
    top_border, bottom_border = int(image.shape[0] * height_neglect), int(
        image.shape[0] * (1 - height_neglect)
    )
    left_border, right_border = int(image.shape[1] * width_neglect), int(
        image.shape[1] * (1 - width_neglect)
    )

    blurred_img = cv2.GaussianBlur(image, (11, 11), 0)

    pre_crop_img = blurred_img[top_border:bottom_border, left_border:right_border]

    sigma = 0.33
    median_intensity = np.median(pre_crop_img)
    lower_thresh = int(max(0, (1.0 - sigma) * median_intensity))
    higher_thresh = int(min(255, (1.0 + sigma) * median_intensity))
    canny_img = cv2.Canny(blurred_img, lower_thresh, higher_thresh)

    sum_rows = np.sum(canny_img == 255, axis=1)
    sum_cols = np.sum(canny_img == 255, axis=0)

    row_thresh = np.percentile(sum_rows, q=85)
    col_thresh = np.percentile(sum_cols, q=85)

    rows = np.where(sum_rows > row_thresh)[0]
    cols = np.where(sum_cols > col_thresh)[0]

    height_redundant = int(pre_crop_img.shape[0] * 0.005)
    width_redundant = int(pre_crop_img.shape[1] * 0.005)

    if rows.size > 0:
        new_top, new_bottom = max(rows[0] - height_redundant, 0), min(
            rows[-1] + height_redundant, pre_crop_img.shape[0]
        )
    else:
        new_top, new_bottom = 0, pre_crop_img.shape[0]
    if cols.size > 0:
        new_left, new_right = max(cols[0] - width_redundant, 0), min(
            cols[-1] + width_redundant, pre_crop_img.shape[1]
        )
    else:
        new_left, new_right = 0, pre_crop_img.shape[1]

    new_top += top_border
    new_bottom += bottom_border
    new_left += left_border
    new_right += right_border

    pre_crop_img2 = blurred_img.copy()
    cv2.rectangle(
        pre_crop_img2,
        (new_top, new_left),
        (new_bottom, new_right),
        (0, 0, 255),
        3,
    )

    # pre_crop_img2 = blurred_img[new_top:new_bottom, new_left:new_right]
    scharr_img = scharr_edge_detection(blurred_img)

    return (
        blurred_img,
        canny_img,
        scharr_img,
        (int(new_top), int(new_left), int(new_bottom), int(new_right)),
    )
