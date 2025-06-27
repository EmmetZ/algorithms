import numpy as np


class StereoSGBM:
    def __init__(self, max_disp_level: int, window_size: int, P1: int, P2: int):
        self.max_disp_level = max_disp_level
        self.window_size = window_size
        self.P1 = P1
        self.P2 = P2
        self.window_h = window_size[0]
        self.window_w = window_size[1]

    def _compute_census_value(self, img: np.ndarray, x_start, y_start) -> np.ndarray:
        """
        compute the census value in one window
        Args:
            img (np.ndarray): image
            x_start (int): start of window in x axis
            y_start (int): start of window in y axis

        Returns:
            1-D array composed of 0s and 1s
        """

        y_offset = self.window_h // 2
        window = img[
            y_start : y_start + self.window_h, x_start : x_start + self.window_w
        ]
        # perform central symmetry operations
        center_symmetry_window = window[::-1, ::-1]
        census_value = np.where((window - center_symmetry_window) > 0, 1, 0)
        return census_value[: y_offset + 1].flatten()

    def _compute_census_value1(self, img: np.ndarray, x_start, y_start) -> np.ndarray:
        """
        compute the census value in one window
        Args:
            img (np.ndarray): image
            x_start (int): start of window in x axis
            y_start (int): start of window in y axis

        Returns:
            1-D array composed of 0s and 1s
        """

        # pick the center pixel
        center = img[y_start + self.window_h // 2, x_start + self.window_w // 2]
        window = img[
            y_start : y_start + self.window_h, x_start : x_start + self.window_w
        ]
        census_value = np.where((window - center) < 0, 0, 1).flatten()
        return census_value

    def _census_transform(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        census transform
        Args:
            left_img (np.ndarray): left image
            right_img (np.ndarray): right image
            window_size (tuple[int, int]): the transform window size (sliding window size)
        Returns:
            transform result of left image and right image
        """

        height, width = left_img.shape

        # need offset, pixel on the border of the image cannot be processed
        x_offset = self.window_w // 2
        y_offset = self.window_h // 2

        # size = window_w * (y_offset + 1)
        size = self.window_w * self.window_h
        left_census_value = np.zeros((height, width, size), dtype=np.int16)
        right_census_value = np.zeros((height, width, size), dtype=np.int16)

        for x in range(x_offset, width - x_offset):
            for y in range(y_offset, height - y_offset):
                x_start = x - x_offset
                y_start = y - y_offset

                # process left image
                left_census_value[y, x] = self._compute_census_value1(
                    left_img, x_start, y_start
                )

                # process right image
                right_census_value[y, x] = self._compute_census_value1(
                    right_img, x_start, y_start
                )
        return left_census_value, right_census_value

    def _compute_cost_volume(
        self,
        left_census_value: np.ndarray,
        right_census_value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        compute cost volume using hamming distance
        Args:
            left_census_value (np.ndarray): left image census transform result
            right_census_value (np.ndarray): right image census transform result
        """

        height, width = left_census_value.shape[:2]
        x_offset = self.window_w // 2
        left_cost_volume = np.zeros((height, width, self.max_disp_level))
        right_cost_volume = np.zeros((height, width, self.max_disp_level))
        tmp_left_census = np.zeros_like(left_census_value, dtype=np.int16)
        tmp_right_census = np.zeros_like(right_census_value, dtype=np.int16)

        for disp in range(self.max_disp_level):
            # compute left cost volume
            # pixel in left image <-> (pixel - disp) in right image
            tmp_right_census[:, x_offset + disp : width - x_offset] = (
                right_census_value[:, x_offset : width - x_offset - disp]
            )
            # compute hamming distance
            hamming_dist = np.bitwise_xor(left_census_value, tmp_right_census).sum(
                axis=2
            )
            # fill left cost volume
            left_cost_volume[:, :, disp] = hamming_dist

            # compute right cost volume
            # pixel in right image <-> (pixel + disp) in left image
            tmp_left_census[:, x_offset : width - x_offset - disp] = left_census_value[
                :, x_offset + disp : width - x_offset
            ]
            # compute hamming distance
            hamming_dist = np.bitwise_xor(tmp_left_census, right_census_value).sum(
                axis=2
            )
            # fill right cost volume
            right_cost_volume[:, :, disp] = hamming_dist

        return left_cost_volume, right_cost_volume

    def _gen_penalties(self) -> np.ndarray:
        """
        generate penalties based on the P1 and P2 values
        """

        p1 = np.full(
            (self.max_disp_level, self.max_disp_level),
            self.P1 - self.P2,
            dtype=np.int16,
        )
        p2 = np.full(
            (self.max_disp_level, self.max_disp_level), self.P2, dtype=np.int16
        )
        p1 = np.triu(p1, k=-1)
        p1 = np.tril(p1, k=1)
        no_penalty = np.identity(self.max_disp_level, dtype=np.int16) * -self.P1
        return p1 + p2 + no_penalty

    def _compute_path_cost(
        self, slice: np.ndarray, penalties: np.ndarray
    ) -> np.ndarray:
        """
        compute cost along one path
        Args:
            slice (np.ndarray): cost volume slice along one path
            penalties (np.ndarray): penalties for disparity changes

        Returns:
            cost along the path
        """

        other_dim, max_disp_level = slice.shape
        costs = np.zeros((other_dim, max_disp_level), dtype=np.int16)
        # initialize the first row of costs, which does not have any previous costs
        costs[0, :] = slice[0, :]

        # start loop from the second row
        for index in range(1, other_dim):
            prev_cost = costs[index - 1, :]
            curr_cost = slice[index, :]
            cost = np.repeat(prev_cost, repeats=max_disp_level).reshape(
                max_disp_level, max_disp_level
            )
            cost += penalties
            cost = cost.min(axis=0) + curr_cost - prev_cost.min()
            costs[index, :] = cost
        return costs

    def _aggregate_cost(self, cost_volume: np.ndarray) -> np.ndarray:
        """
        Aggregate cost volume along 4 paths (left, right, up, down) using penalties.
        Args:
            cost_volume (np.ndarray): cost volume with shape (H, W, D)
        """
        height, width = cost_volume.shape[:2]

        penalties = self._gen_penalties()
        up_down_cost = np.zeros_like(cost_volume, dtype=np.int16)
        down_up_cost = np.zeros_like(cost_volume, dtype=np.int16)

        for w in range(width):
            up_down_slice = cost_volume[:, w, :]
            down_up_slice = np.flip(up_down_slice, axis=0)

            up_down_cost[:, w, :] = self._compute_path_cost(up_down_slice, penalties)
            down_up_cost[:, w, :] = np.flip(
                self._compute_path_cost(down_up_slice, penalties), axis=0
            )

        left_right_cost = np.zeros_like(cost_volume, dtype=np.int16)
        right_left_cost = np.zeros_like(cost_volume, dtype=np.int16)

        for h in range(height):
            left_right_slice = cost_volume[h, :, :]
            right_left_slice = np.flip(left_right_slice, axis=0)

            left_right_cost[h, :, :] = self._compute_path_cost(
                left_right_slice, penalties
            )
            right_left_cost[h, :, :] = np.flip(
                self._compute_path_cost(right_left_slice, penalties), axis=0
            )

        aggregate_cost = np.concatenate(
            (
                left_right_cost[..., np.newaxis],
                right_left_cost[..., np.newaxis],
                up_down_cost[..., np.newaxis],
                down_up_cost[..., np.newaxis],
            ),
            axis=3,
        )
        return aggregate_cost.sum(axis=3)

    def _select_disparity(self, cost_volume: np.ndarray) -> np.ndarray:
        """
        Select the disparity with the minimum cost for each pixel.
        Args:
            cost_volume (np.ndarray): aggregated cost volume with shape (H, W, D)

        Returns:
            disparity map with shape (H, W)
        """
        return np.argmin(cost_volume, axis=2)

    def compute(self, left_img, right_img) -> tuple[np.ndarray, np.ndarray]:
        left_census_value, right_census_value = self._census_transform(
            left_img, right_img
        )

        left_cost_volume, right_cost_volume = self._compute_cost_volume(
            left_census_value, right_census_value
        )

        left_agg_cost = self._aggregate_cost(left_cost_volume)
        right_agg_cost = self._aggregate_cost(right_cost_volume)

        left_disp = self._select_disparity(left_agg_cost)
        right_disp = self._select_disparity(right_agg_cost)

        return left_disp, right_disp
