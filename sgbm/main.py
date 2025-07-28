from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from sgbm import StereoSGBM


if __name__ == "__main__":
    left_img = np.array(
        Image.open("./input/2003/cones/im2.png").convert("L"),
        dtype=np.int16,
    )
    right_img = np.array(
        Image.open("./input/2003/cones/im6.png").convert("L"),
        dtype=np.int16,
    )
    print("Left image shape:", left_img.shape)
    print("Right image shape:", right_img.shape)
    assert left_img.shape == right_img.shape, (
        "Left and right images must have the same shape"
    )

    window_size = (5, 5)
    max_disp_level = 64
    P1 = 10
    P2 = 100

    stereo = StereoSGBM(
        max_disp_level=max_disp_level, window_size=window_size, P1=P1, P2=P2
    )

    left_disp, right_disp = stereo.compute(left_img, right_img)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(left_disp, cmap="gray")
    plt.title("Left Disparity Map")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(right_disp, cmap="gray")
    plt.title("Right Disparity Map")
    plt.axis("off")
    plt.show()
