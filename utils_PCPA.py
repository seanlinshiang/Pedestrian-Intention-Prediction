import sys
import PIL
import os
import pickle
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import accuracy_score


def jitter_bbox(img, bbox, mode, ratio):
    """
    Jitters the position or dimensions of the bounding box.
    Args:
        img_path: The to the image
        bbox: The bounding box to be jittered
        mode: The mode of jitterring. Options are,
          'same' returns the bounding box unchanged
          'enlarge' increases the size of bounding box based on the given ratio.
          'random_enlarge' increases the size of bounding box by randomly sampling a value in [0,ratio)
          'move' moves the center of the bounding box in each direction based on the given ratio
          'random_move' moves the center of the bounding box in each direction by randomly
                        sampling a value in [-ratio,ratio)
        ratio: The ratio of change relative to the size of the bounding box.
           For modes 'enlarge' and 'random_enlarge'
           the absolute value is considered.
    Return:
        Jitterred bounding boxes
    """

    assert mode in ["same", "enlarge", "move", "random_enlarge", "random_move"], (
        "mode %s is invalid." % mode
    )

    if mode == "same":
        return bbox

    if mode in ["random_enlarge", "enlarge"]:
        jitter_ratio = abs(ratio)
    else:
        jitter_ratio = ratio

    if mode == "random_enlarge":
        jitter_ratio = np.random.random_sample() * jitter_ratio
    elif mode == "random_move":
        # for ratio between (-jitter_ratio, jitter_ratio)
        # for sampling the formula is [a,b), b > a,
        # random_sample * (b-a) + a
        jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

    jit_boxes = []
    for b in bbox:
        bbox_width = b[2] - b[0]
        bbox_height = b[3] - b[1]

        width_change = bbox_width * jitter_ratio
        height_change = bbox_height * jitter_ratio

        if width_change < height_change:
            height_change = width_change
        else:
            width_change = height_change

        if mode in ["enlarge", "random_enlarge"]:
            b[0] = b[0] - width_change // 2
            b[1] = b[1] - height_change // 2
        else:
            b[0] = b[0] + width_change // 2
            b[1] = b[1] + height_change // 2

        b[2] = b[2] + width_change // 2
        b[3] = b[3] + height_change // 2

        # Checks to make sure the bbox is not exiting the image boundaries
        b = bbox_sanity_check([img.shape[1], img.shape[0]], b)

        jit_boxes.append(b)
    # elif crop_opts['mode'] == 'border_only':
    return jit_boxes


def squarify(bbox, squarify_ratio, img_width):
    """
    Changes the dimensions of a bounding box to a fixed ratio
    Args:
        bbox: Bounding box
        squarify_ratio: Ratio to be changed to
        img_width: Image width
    Return:
        Squarified boduning boxes
    """
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    bbox[0] = bbox[0] - width_change / 2
    bbox[2] = bbox[2] + width_change / 2
    # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
    if bbox[0] < 0:
        bbox[0] = 0

    # check whether the new bounding box goes beyond image boarders
    # If this is the case, the bounding box is shifted back
    if bbox[2] > img_width:
        # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
        bbox[0] = bbox[0] - bbox[2] + img_width
        bbox[2] = img_width
    return bbox


def img_pad(img, mode="warp", size=224):
    """
    Pads and/or resizes a given image
    Args:
        img: The image to be coropped and/or padded
        mode: The type of padding or resizing. Options are,
            warp: crops the bounding box and resize to the output size
            same: only crops the image
            pad_same: maintains the original size of the cropped box  and pads with zeros
            pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
                        the desired output size in that direction while maintaining the aspect ratio. The rest
                        of the image is	padded with zeros
            pad_fit: maintains the original size of the cropped box unless the image is bigger than the size
                    in which case it scales the image down, and then pads it
        size: Target size of image
    Return:
        Padded image
    """
    assert mode in ["same", "warp", "pad_same", "pad_resize", "pad_fit"], (
        "Pad mode %s is invalid" % mode
    )
    image = np.copy(img)
    if mode == "warp":
        warped_image = cv2.resize(img, (size, size))
        return warped_image
    elif mode == "same":
        return image
    elif mode in ["pad_same", "pad_resize", "pad_fit"]:
        img_size = image.shape[:2][::-1]  # original size is in (height, width)
        ratio = float(size) / max(img_size)
        if mode == "pad_resize" or (
            mode == "pad_fit" and (img_size[0] > size or img_size[1] > size)
        ):
            img_size = tuple([int(img_size[0] * ratio), int(img_size[1] * ratio)])
            image = cv2.resize(image, img_size)
        padded_image = np.zeros((size, size) + (image.shape[-1],), dtype=img.dtype)
        w_off = (size - img_size[0]) // 2
        h_off = (size - img_size[1]) // 2
        padded_image[
            h_off : h_off + img_size[1], w_off : w_off + img_size[0], :
        ] = image
        return padded_image


def bbox_sanity_check(img_size, bbox):
    """
    Checks whether  bounding boxes are within image boundaries.
    If this is not the case, modifications are applied.
    Args:
        img_size: The size of the image
        bbox: The bounding box coordinates
    Return:
        The modified/original bbox
    """
    img_width, img_heigth = img_size
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_width:
        bbox[2] = img_width - 1
    if bbox[3] >= img_heigth:
        bbox[3] = img_heigth - 1
    return bbox
