import numpy as np
import math
from .config import IMG_WIDTH, IMG_HEIGHT, FOV, SIM


class Matrix:
    """
    Matrix class for handling camera projection and transformation matrices.
    """

    def __init__(self, vehicle="verifier", yaw=0, pitch=0, roll=0, camera_location=[], distance=50):
        """
        Initialize Matrix object.

        Args:
            vehicle (str): Vehicle name.
            yaw (float): Yaw angle in degrees.
            pitch (float): Pitch angle in degrees.
            roll (float): Roll angle in degrees.
            camera_location (list): Camera location [x, y, z].
            distance (float): Distance from camera to object.
        """
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.camera_location = [0, 0, 1.5953] if SIM else [0, 0, 1.3716]
        self.vehicle = vehicle
        self.int = self.compute_intrinsic()
        self.ext = self.compute_extrinsic()

    def point_to_image(self, distance, left=False, delta=5, theta=0):
        """
        Convert a 3D point to image coordinates.

        Args:
            distance (float): Distance from camera to point.
            left (bool): Flag indicating if the point is on the left side of the camera.
            delta (float): Distance from the camera center to the point.
            theta (float): Angle in degrees.

        Returns:
            list: Image coordinates [x, y].
        """
        multiplier = -1 if left else 1
        half_fov_rad = math.radians(FOV / 2)
        fov_width = 2 * (math.tan(half_fov_rad) * distance / 2)
        fov_point = np.array([distance / 2, 0, 0]) + np.array([0, multiplier * fov_width / 2, 0])
        fov_point_hom = np.append(fov_point, 1)
        point_camera = np.dot(self.ext, fov_point_hom)
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
        estimated_point_img = np.dot(self.int, point_camera)
        estimated_point_img[0] /= estimated_point_img[2]
        estimated_point_img[1] /= estimated_point_img[2]

        x = estimated_point_img[0]
        y = estimated_point_img[1]
        return [x, y]

    
    # ext_matrix: 4x4 matrix
    # 3x3 rotation | 3x1 translation vector
    # 1x4 [0 0 0 1]
    def get_horizon(self, distance, delta=5, theta=0):
        """
        Compute the horizon line in the image.

        Args:
            distance (float): Distance from camera to horizon.
            delta (float): Distance from the camera center to the horizon.
            theta (float): Angle in degrees.
        """
        self.left = self.point_to_image(distance, left=True, theta=theta, delta=delta)
        self.right = self.point_to_image(distance, left=False, theta=theta, delta=delta)

    def tune(self, follower):
        """
        Tune the pitch angle to match the follower's position.

        Args:
            follower (object): Follower object.
        """
        y = follower.y  # obtained from image
        pitch = 22
        threshold = 22
        found = False
        old_y_hat = 0
        self.pitch = 22
        self.int = self.int if SIM else self.compute_intrinsic()
        while self.pitch > -threshold:
            self.ext = self.compute_extrinsic()
            point = np.dot(self.ext, np.array([follower.distance, 0, 0, 1]))
            point = [point[1], -point[2], point[0]]
            est = np.dot(self.int, point)
            y_hat = est[1] / est[2]
            error = y_hat - y
            if abs(error) < 1:
                found = True
                break
            delta = (y_hat - y) / 27.98551577143371
            self.pitch -= delta
        if not found:
            self.pitch = None
        self.int = self.int if SIM else self.compute_intrinsic()
        self.ext = self.compute_extrinsic()

    def compute_intrinsic(self):
        """
        Compute the intrinsic matrix.

        Returns:
            numpy.ndarray: Intrinsic matrix.
        """
        focal = IMG_WIDTH / (2 * np.tan(FOV * np.pi / 360.0))
        return np.array([[focal, 0, IMG_WIDTH / 2], [0, focal, IMG_HEIGHT / 2], [0, 0, 1]])

    def compute_extrinsic(self):
        """
        Compute the extrinsic matrix.

        Returns:
            numpy.ndarray: Extrinsic matrix.
        """
        cy = math.cos(math.radians(self.yaw))
        sy = math.sin(math.radians(self.yaw))
        cp = math.cos(math.radians(self.pitch))
        sp = math.sin(math.radians(self.pitch))
        cr = math.cos(math.radians(self.roll))
        sr = math.sin(math.radians(self.roll))

        t_x = self.camera_location[0]
        t_y = self.camera_location[1]
        t_z = self.camera_location[2]

        return np.linalg.inv(np.array([[cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, t_x],
                                       [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, t_y],
                                       [sp, -cp * sr, cp * cr, t_z],
                                       [0, 0, 0, 1]]))
