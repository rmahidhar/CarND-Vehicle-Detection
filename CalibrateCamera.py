import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import glob


class Calibrate(object):

    def __init__(self, calibration_image_files, pattern_size=(9, 6)):
        #images = glob.glob("camera_cal/*.jpg")
        # camera matrix returned by calibrate camera
        self._camera_matrix = None
        # distortion coefficients returned by calibrate camera
        self._dist_coeffs = None
        self._calibration_success = None
        self._calibrate(calibration_image_files, pattern_size)

    def _calibrate(self, image_files, pattern_size):
        # 3D points in real world space
        object_points = []
        # 2D points in image plane
        image_points = []

        # Prepare object points, (x, y, z) like (0,0,0), (1,0,0) ... (7,5,0)
        # The z co-ordinate is 0 because image plane is 2D and flat object.
        pattern = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        # x, y coordinates
        pattern[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        gray_image = None
        for i, file in enumerate(image_files):
            image = matplotlib.image.imread(file)
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray_image, pattern_size, None)
            # If corners are found, add objects points, image points
            if ret:
                image_points.append(corners)
                object_points.append(pattern)
                # Draw and display the corners
                image = cv2.drawChessboardCorners(image, pattern_size, corners, ret)
                plt.figure(i+1)
                plt.imshow(image)
                plt.title(file)
            else:
                print("findChessBoardCorners for {} failed".format(file))
                self._calibration_success = False

        self._calibration_success, self._camera_matrix, self._dist_coeffs, _, _ = \
            cv2.calibrateCamera(object_points,
                                image_points,
                                gray_image.shape[::-1],
                                None,
                                None)

    def __call__(self, image):
        return self.undistort(image)

    def undistort(self, image):
        return cv2.undistort(image, self._camera_matrix,
                             self._dist_coeffs,
                             None,
                             self._camera_matrix)





