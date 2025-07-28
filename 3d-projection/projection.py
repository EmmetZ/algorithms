import numpy as np
import torch
from torch.nn import functional as F


class BackProjection:
    def __init__(self, width: int, height: int):
        """
        Initializes the BackProjection class with the given width and height.
        Args:
            width (int): The width of the image.
            height (int): The height of the image.
        """
        self.width = width
        self.height = height

        self.init_coords()

    def init_coords(self):
        """
        Initializes the pixel coordinates for back projection.
        """
        # Create a grid of pixel coordinates
        mesh = np.meshgrid(range(self.width), range(self.height), indexing="xy")

        # shape: (2, H, W)
        id_coords = np.stack(mesh, axis=0).astype(np.float32)

        # stack the x and y coordinates to create a 2D array of pixel coordinates
        # shape: (2, H * W)
        pix_coords = id_coords.reshape(2, -1)

        # shape: (1, H * W)
        ones = np.ones((1, self.width * self.height), dtype=np.float32)

        # 3d coordinates: (x, y, 1)
        # shape: (3, H * W)
        self.pix_coords = np.concatenate((pix_coords, ones), axis=0)

    def back_project(self, depth: np.ndarray, inv_K: np.ndarray):
        """
        Back projects the pixel coordinates using the depth map and inverse camera matrix.
        Args:
            depth (np.ndarray): Depth map of shape (H, W).
            inv_K (np.ndarray): Inverse camera matrix of shape (3, 3).
        """
        # shape: (3, H * W)
        pix_coords_3d = np.matmul(inv_K, self.pix_coords)

        # scale by depth
        pix_coords_3d *= depth.reshape(1, -1)

        # add a row of ones for homogeneous coordinates
        pix_coords_3d = np.vstack(
            (pix_coords_3d, np.ones((1, self.width * self.height), dtype=np.float32))
        )

        return pix_coords_3d


class Project3D:
    def __init__(self, width: int, height: int):
        """
        Initializes the Project3D class with the given width and height.
        Args:
            width (int): The width of the image.
            height (int): The height of the image.
        """
        self.width = width
        self.height = height

    def project(
        self, coords_3d: np.ndarray, P: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        """
        Projects 3D coordinates to 2D pixel coordinates using the camera matrix.
        Args:
            coords_3d (np.ndarray): 3D coordinates of shape (4, N).
            P (np.ndarray): Camera projection matrix of shape (3, 4).
            T (np.ndarray): Transformation matrix of shape (4, 4).
        Returns:
            np.ndarray: Projected pixel coordinates of shape (2, H, W).
        """

        P_trans = np.matmul(P, T)[:3]
        coords_2d: np.ndarray = np.matmul(P_trans, coords_3d)

        # normalize by z coordinate
        # +1e-8 to avoid division by zero
        pix_coords = coords_2d[:2] / (coords_2d[2].reshape(1, -1) + 1e-8)

        return pix_coords.reshape(2, self.height, self.width)


def grid_sample(img: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """
    Interpolates pixel values from the image at the given coordinates.
    Args:
        img (np.ndarray): Input image of shape (H, W, C).
        coords (np.ndarray): Coordinates to sample from, of shape (2, H, W).
    Returns:
        np.ndarray: Interpolated pixel values.
    """

    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    coords = torch.tensor(coords).float().permute(1, 2, 0).unsqueeze(0)  # (1, H, W, 2)
    width, height = img.shape[3], img.shape[2]
    coords[..., 0] /= width - 1
    coords[..., 1] /= height - 1
    coords = (coords - 0.5) * 2

    # Use grid_sample for interpolation
    img_reproj = F.grid_sample(
        img,
        coords,
        mode="bilinear",
        padding_mode="border",
    )

    np_img_reproj = img_reproj.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, C)
    return np_img_reproj
