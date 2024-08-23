import os
import numpy as np
import cv2
from staintools.preprocessing.input_validation import is_uint8_image


def validate_file_fmt(source, file_name):
    if os.path.isdir(os.path.join(source, file_name)):
        return None
    for fmt in ['ndpi', 'svs', 'tiff', 'tif']: # *note* tiff must be pior to tif!!!
        if fmt in file_name:
            return fmt
    return None


def standardize(I, p=255, percentile=95):
    """
    Source code adapted from:
    https://github.com/Peter554/StainTools/blob/master/staintools
    staintools.LuminosityStandardizer.standardize
    
    Transform image I to standard brightness.
    Modifies the luminosity channel such that a fixed percentile is saturated.

    :param I: Image uint8 RGB.
    :param percentile: Percentile for luminosity saturation. At least (100 - percentile)% of pixels should be fully luminous (white).
    :return: Image uint8 RGB with standardized brightness.
    """
    assert is_uint8_image(I), "Image should be RGB uint8."
    I_LAB = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    L_float = I_LAB[:, :, 0].astype(float)
    if p == 255:
        p = np.percentile(L_float, percentile)
    I_LAB[:, :, 0] = np.clip(255 * L_float / p, 0, 255).astype(np.uint8)
    I = cv2.cvtColor(I_LAB, cv2.COLOR_LAB2RGB)
    return I, p
