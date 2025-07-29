import numpy as np
from projection import BackProjection, Project3D, grid_sample
from pathlib import Path
from dataset import KITTIDataset
from matplotlib import pyplot as plt
from skimage.restoration import inpaint

if __name__ == "__main__":
    root = Path(__file__).parent

    dataset_root = root / "data/2011_09_26_drive_0048_sync"
    assert dataset_root.exists()

    dataset = KITTIDataset(dataset_root)
    data = dataset.get(1)

    left_img = data.left_img
    right_img = data.right_img
    left_depth = data.left_depth
    width, height = left_img.shape[1], left_img.shape[0]

    K = data.left_P[:3, :3]
    inv_K = np.linalg.pinv(K)

    # interpolate missing depth values
    left_depth_inpaint = inpaint.inpaint_biharmonic(left_depth, left_depth < 1)

    # Back-project depth to 3D points
    back_proj = BackProjection(width, height)
    coords_3d = back_proj.back_project(left_depth_inpaint, inv_K)

    # Project 3D points to the right camera's view
    proj = Project3D(width, height)
    T_left_to_right = np.eye(4)

    # The baseline is the difference in the x-translation component of the projection matrices
    baseline = np.abs(
        data.right_P[0, 3] / data.right_P[0, 0] - data.left_P[0, 3] / data.left_P[0, 0]
    )
    print(f"baseline: {baseline}")

    # x-translation from left to right camera
    T_left_to_right[0, 3] = -baseline

    # Project the 3D coordinates to the right image
    img_coords = proj.project(coords_3d, data.left_P, T_left_to_right)

    img_reproj = grid_sample(right_img, img_coords)
    diff = img_reproj - left_img

    fig, axes = plt.subplots(2, 2, figsize=(12, 5))

    axes[0, 0].imshow(left_img)
    axes[0, 0].set_title("Left Image")
    axes[0, 1].imshow(right_img)
    axes[0, 1].set_title("Right Image")
    axes[1, 0].imshow(img_reproj)
    axes[1, 0].set_title("Reprojected Right Image to Left View")
    axes[1, 1].imshow(left_depth, cmap="gray")
    axes[1, 1].set_title("Left Depth (Velodyne)")

    for ax in axes.flatten():
        ax.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.savefig("reprojection_result.png", dpi=600)
    plt.show()
