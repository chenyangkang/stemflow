import random

import matplotlib
import numpy as np


def generate_soft_color() -> np.ndarray:
    """generating random soft colors

    Returns:
        rgb (tuple): a rgb color
    """

    # Generate random hue (between 0 and 1)
    hue = random.random()

    # Generate a lower saturation value for a softer color
    saturation = random.uniform(0.4, 0.7)

    # Generate a lower brightness value for a softer color
    brightness = random.uniform(0.6, 0.9)

    rgb = matplotlib.colors.hsv_to_rgb([hue, saturation, brightness])

    return rgb
