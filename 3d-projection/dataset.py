from pathlib import Path
import numpy as np
from dataclasses import dataclass
from collections import Counter
from typing import Literal
import skimage
from PIL import Image


@dataclass
class KITTIData:
    left_img: np.ndarray
    right_img: np.ndarray
    left_depth: np.ndarray
    right_depth: np.ndarray
    left_P: np.ndarray
    right_P: np.ndarray


class KITTIDataset:
    """
    A class to handle the KITTI dataset for 3D projection.
    """

    def __init__(self, root_dir: Path):
        """
        Initializes the KITTI dataset with the root directory.
        Args:
            root_dir (Path): The root directory of the KITTI dataset.
        """
        self.root_dir = root_dir
        self.full_res_shape = (375, 1242)

        calib_dir = self.root_dir / "calib"
        self.cam_to_cam_calib = self.read_calib_file(calib_dir / "calib_cam_to_cam.txt")
        self.velo_to_cam_calib = self.read_calib_file(
            calib_dir / "calib_velo_to_cam.txt"
        )
        self.left_P = self.cam_to_cam_calib["P_rect_02"].reshape(3, 4)
        self.right_P = self.cam_to_cam_calib["P_rect_03"].reshape(3, 4)

    def get(self, idx: int, do_flip=False) -> KITTIData:
        """
        get one set of data from dataset
        Args:
            idx (int): Index of the data to retrieve.
            do_flip (bool): Whether to flip the images and depth maps horizontally.
        """
        output = KITTIData(
            left_img=self.get_image(idx, "left"),
            right_img=self.get_image(idx, "right"),
            left_depth=self.get_depth(idx, "left", do_flip=do_flip),
            right_depth=self.get_depth(idx, "right", do_flip=do_flip),
            left_P=self.left_P.copy(),
            right_P=self.right_P.copy(),
        )

        return output

    def get_image(
        self,
        idx: int,
        side: Literal["left", "right"] = "left",
    ) -> np.ndarray:
        img_idx = f"{idx:010d}.png"
        side_idx = "02" if side == "left" else "03"
        img_path = self.root_dir / f"image_{side_idx}" / "data" / img_idx
        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32) / 255.0
        return img

    def get_depth(
        self,
        idx: int,
        side: Literal["left", "right"] = "left",
        do_flip: bool = False,
    ):
        """
        Get depth map for a specific index.
        Args:
            idx (int): Index of the data to retrieve.
            side (Literal["left", "right"]): Side of the depth map to retrieve.
            do_flip (bool): Whether to flip the depth map horizontally.
        """
        velo_path = self.root_dir / "velodyne_points" / "data" / f"{idx:010d}.bin"

        side = 2 if side == "left" else 3
        depth_gt = self.gen_depth_map(
            velo_path,
            camera_id=side,
        )

        # resize depth map to full resolution
        depth_gt = skimage.transform.resize(
            depth_gt,
            self.full_res_shape,
            order=0,
            preserve_range=True,
            mode="constant",
        )

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def gen_depth_map(
        self,
        velo_path: Path,
        camera_id: int = 2,
        velo_depth: bool = False,
    ) -> np.ndarray:
        """
        Generate depth map from velodyne points and calibration data.
        Args:
            velo_path (Path): Path to the velodyne points file.
            camera_id (int, optional): Camera ID to use for projection. 2 is left, 3 is right. Default is 2.
            velo_depth (bool): Whether to use velodyne depth.
        """

        # read calibration files

        # build transformation matrix from velodyne coordinates to camera coordinates
        # matrix is 4x4: [R | T]
        #                [0 | 1]
        velo_to_cam = np.hstack(
            (
                self.velo_to_cam_calib["R"].reshape(3, 3),
                self.velo_to_cam_calib["T"].reshape(3, 1),
            )
        )
        velo_to_cam = np.vstack((velo_to_cam, np.array([0, 0, 0, 1.0])))

        # read image shape
        W, H = self.cam_to_cam_calib[f"S_rect_0{camera_id}"].astype(np.int32)

        # build transformation matrix from camera coordinates to rectified camera coordinates
        # matrix is 4x4: [R_rect | 0]
        #                [0      | 1]
        R_cam_to_rect = np.eye(4)
        R_cam_to_rect[:3, :3] = self.cam_to_cam_calib["R_rect_00"].reshape(3, 3)

        # build projection matrix from rectified
        # camera coordinates (3D) to image coordinates (2D)
        P_rect = self.cam_to_cam_calib[f"P_rect_0{camera_id}"].reshape(3, 4)

        # combine all transformations matrices
        # P_velo_to_img = P_rect * R_cam_to_rect * velo_to_cam
        P_velo_to_img = np.dot(np.dot(P_rect, R_cam_to_rect), velo_to_cam)

        # load velodyne points
        velo = self.load_velodyne_points(velo_path)
        # filter out points with negative x coordinate (i.e., behind the camera)
        velo = velo[velo[:, 0] > 0, :]

        # project velodyne points to image coordinates
        # [x, y, z, 1] => [u * Z, v * Z, Z]
        velo_pts_img = np.dot(P_velo_to_img, velo.T).T
        # compute pixel coordinates in image coordinates
        velo_pts_img[:, :2] = velo_pts_img[:, :2] / velo_pts_img[:, 2][..., np.newaxis]

        # if velo_depth is True, use the x coordinate of the velodyne points as depth
        if velo_depth:
            velo_pts_img[:, 2] = velo[:, 0]

        # check if points are in image bounds
        # use minus 1 to get the exact same value as KITTI matlab code
        # (python uses 0-based indexing)
        velo_pts_img[:, :2] = np.round(velo_pts_img[:, :2]) - 1
        valid_indice = (velo_pts_img[:, 0] >= 0) & (velo_pts_img[:, 1] >= 0)
        valid_indice = (
            valid_indice & (velo_pts_img[:, 0] < W) & (velo_pts_img[:, 1] < H)
        )
        velo_pts_img = velo_pts_img[valid_indice, :]

        # project points to depth map
        depth = np.zeros((H, W))
        # depth[y, x] = z
        depth[
            velo_pts_img[:, 1].astype(np.int32), velo_pts_img[:, 0].astype(np.int32)
        ] = velo_pts_img[:, 2]

        # flatten 2D coordinates to linear indices
        # make finding duplicate indices easier
        # inds = y * (W - 1) + x - 1
        inds = self.flatten_2d_coords(W, velo_pts_img[:, 1], velo_pts_img[:, 0])

        # find duplicate indices
        dupe_inds = [item for item, count in Counter(inds).items() if count > 1]

        # for each duplicate index, take the minimum depth value (closest to the camera)
        for dd in dupe_inds:
            # find the index of first occurrence of the duplicate index
            pts_idx = np.where(inds == dd)[0]
            # get the x and y coordinates of all points with the same index
            x_loc = int(velo_pts_img[pts_idx[0], 0])
            y_loc = int(velo_pts_img[pts_idx[0], 1])
            # set the depth value to the minimum depth value of all points with the same index
            depth[y_loc, x_loc] = velo_pts_img[pts_idx, 2].min()

        # set negative depth values to 0
        depth[depth < 0] = 0

        return depth

    def read_calib_file(self, path: Path) -> dict:
        """
        Read KITTI calibration file (from https://github.com/hunse/kitti)
        """
        float_chars = set("0123456789.e+- ")
        data = {}
        with path.open("r") as f:
            for line in f.readlines():
                key, value = line.split(":", 1)
                value = value.strip()
                data[key] = value
                if float_chars.issuperset(value):
                    # try to cast to float array
                    try:
                        data[key] = np.array(list(map(float, value.split(" "))))
                    except ValueError:
                        # casting error: data[key] already eq. value, so pass
                        pass
        return data

    def load_velodyne_points(self, velo_path: Path) -> np.ndarray:
        """
        Load 3D point cloud from KITTI file format
        (adapted from https://github.com/hunse/kitti)
        Args:
            velo_path (Path): Path to the velodyne points file.
        Returns:
            np.ndarray: Velodyne points as a numpy array.
        """
        points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
        points[:, 3] = 1.0  # set intensity to 1.0
        return points

    def flatten_2d_coords(
        self, depth_width: int, y_coords: np.ndarray, x_coords: np.ndarray
    ) -> np.ndarray:
        """
        Flatten 2D coordinates to linear indices.

        Args:
            depth_width (int): width of the depth map.
            y_coords (np.ndarray): Y coordinates.
            x_coords (np.ndarray): X coordinates.

        Returns:
            np.ndarray: Flattened indices.
        """
        return y_coords * (depth_width - 1) + x_coords - 1
