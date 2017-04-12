import cv2
import numpy as np


def absolute_sobel_threshold(image, kernel=3, orient='x', thresh=(0, 255)):
    # Calculate x and y gradients and take the absolute value
    if orient == 'x':
        absolute = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel))
    elif orient == 'y':
        absolute = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel))
    # Rescale back to 8 bit integer
    absolute = np.uint8(255 * absolute / np.max(absolute))
    # Create a binary image of ones where threshold is met, zero otherwise
    binary_output = np.zeros_like(absolute)
    binary_output[(absolute >= thresh[0]) & (absolute <= thresh[1])] = 1
    return binary_output


def magnitude_sobel_threshold(image, kernel=3, thresh=(0, 255)):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel)
    # Calculate the gradient magnitude
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale back to 8 bit integer
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))
    # Create a binary image of ones where threshold is met, zero otherwise
    binary_output = np.zeros_like(magnitude)
    binary_output[(magnitude >= thresh[0]) & (magnitude <= thresh[1])] = 1
    return binary_output


def direction_sobel_threshold(image, kernel=3, thresh=(0, np.pi / 2)):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel)
    # Take the absolute value of the gradient direction,
    direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary image of ones where threshold is met, zero otherwise
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output


#def color_threshold(image, thresh=(0, 255)):
#    binary_output = np.zeros_like(image)
#    binary_output[(image > thresh[0]) & (image <= thresh[1])] = 1
#    return binary_output

def color_threshold(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    yellow_min = np.array([15, 100, 120], np.uint8)
    yellow_max = np.array([80, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(image, yellow_min, yellow_max)

    white_min = np.array([0, 0, 200], np.uint8)
    white_max = np.array([255, 30, 255], np.uint8)
    white_mask = cv2.inRange(image, white_min, white_max)

    binary_output = np.zeros_like(image[:, :, 0])
    binary_output[((yellow_mask != 0) | (white_mask != 0))] = 1

    filtered = image
    filtered[((yellow_mask == 0) & (white_mask == 0))] = 0

    return binary_output

class GradientPipeline(object):
    def __init__(self):
        pass
        self.kernel = 3

        self.direction_threshold = (0.7, 1.3)
        self.magnitude_threshold = (50, 255)
        self.absolute_threshold = (100, 200)
        self.color_threshold = (170, 255)

        self.s_channel = 2
        self.r_channel = 0

    def __call__(self, image, stacked=False):
        #hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
        #chan = hls[:, :, self.s_channel]
        # Get red channel from the image
        chan = image[:, :, self.r_channel]
        # apply sobel threshold on saturation channel
        absx = absolute_sobel_threshold(chan, kernel=self.kernel, orient='x', thresh=self.absolute_threshold)
        absy = absolute_sobel_threshold(chan, kernel=self.kernel, orient='y', thresh=self.absolute_threshold)
        magnitude = magnitude_sobel_threshold(chan, kernel=self.kernel, thresh=self.magnitude_threshold)
        direction = direction_sobel_threshold(chan, kernel=self.kernel, thresh=self.direction_threshold)
        gradient = np.zeros_like(chan)
        gradient[((absx == 1) & (absy == 1)) | ((magnitude == 1) & (direction == 1))] = 1
        #gradient[((magnitude == 1) & (direction == 1))] = 1
        # apply color threshold mask
        #color = color_threshold(chan, thresh=self.color_threshold)
        color = color_threshold(image)
        if stacked:
            return np.dstack((np.zeros_like(chan), gradient, color))
        else:
            binary_output = np.zeros_like(chan)
            binary_output[(gradient == 1) | (color == 1)] = 1
            return binary_output


class Warper:

    def __init__(self):
        src = np.float32([
            [580, 460],
            [700, 460],
            [1040, 680],
            [260, 680],
        ])

        dst = np.float32([
            [260, 0],
            [1040, 0],
            [1040, 720],
            [260, 720],
        ])

        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def __call__(self, image, unwarp=False):
        if unwarp:
            return self.unwarp(image)
        else:
            return self.warp(image)

    def warp(self, image):
        return cv2.warpPerspective(
            image,
            self.M,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR
        )

    def unwarp(self, image):
        return cv2.warpPerspective(
            image,
            self.M_inv,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR
        )


class GradientWarpPipeline(object):

    def __call__(self, image, stacked=False):
        gradient = GradientPipeline()(image, stacked)
        return Warper()(gradient)

