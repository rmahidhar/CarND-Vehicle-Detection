import cv2
import numpy as np
import Pipeline
import Line
from ImagePlotter import ImagePlotter


class LaneFinder(object):
    MARGIN = 100
    MIN_PIXELS = 50

    class Window(object):
        def __init__(self, x, low_y, high_y):
            self.x = x
            self.low_y = low_y
            self.high_y = high_y
            self.left_x = self.x - LaneFinder.MARGIN
            self.right_x = self.x + LaneFinder.MARGIN
            self.mean_x = x

        def pixels_indices(self, nonzero, x=None):
            if x is not None:
                self.x = x
                self.left_x = self.x - LaneFinder.MARGIN
                self.right_x = self.x + LaneFinder.MARGIN

            y = nonzero[0]
            x = nonzero[1]
            indices = ((y >= self.low_y) & (y < self.high_y) &
            (x >= self.left_x) & (x < self.right_x)).nonzero()[0]
            if len(indices) > LaneFinder.MIN_PIXELS:
                self.mean_x = np.int(np.mean(x[indices]))
            else:
                self.mean_x = self.x
            return indices

        def coordinates(self):
            return (self.left_x, self.low_y), (self.right_x, self.high_y)

        def __str__(self):
            return "{}".format(self.cooridnates())

    def __init__(self, image):

        self._num_windows = 9
        self._height, self._width, _ = image.shape
        self._window_height = np.int(self._height/self._num_windows)

        self._left_lane = None
        self._right_lane = None
        # Set the width of the window +/- margin
        self._margin = 100
        # Set minimum number of pixels found to recenter window
        self._min_pixels = 50
        # Empty lists for left and right lane pixel indices
        self._left_lane_windows = []
        self._right_lane_windows = []
        self._current_left_x = []
        self._current_left_y = []
        self._current_right_x = []
        self._current_right_y = []
        self._init(image)

    def _init(self, frame):
        # pipeline returns bird eye view perspective image, after applying sobel gradient
        # and color thresholds. The return image has pixel values of 0 and 1
        image = Pipeline.GradientWarpPipeline()(frame)

        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[self._height/2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        midpoint = np.int(histogram.shape[0]/2)
        left_x = np.argmax(histogram[:midpoint])
        right_x = np.argmax(histogram[midpoint:]) + midpoint

        left_lane_indices = []
        right_lane_indices = []

        # Get nonzero pixels in the image. The nonzero() return a tuple of x, y points
        # that have nonzero value
        nonzero = image.nonzero()

        # Divide the image into 9 windows for lane detection.
        for window in range(self._num_windows):
            # Identify window boundaries in x and y (and right and left)
            # y value increases from top to bottom, x values increases from left to right.
            low_y = image.shape[0] - (window + 1) * self._window_height
            high_y = image.shape[0] - window * self._window_height

            left_window = LaneFinder.Window(left_x, low_y, high_y)
            right_window = LaneFinder.Window(right_x, low_y, high_y)

            self._left_lane_windows.append(left_window)
            self._right_lane_windows.append(right_window)

            # Get nonzero pixel indices within the window
            left_nonzero_indices = left_window.pixels_indices(nonzero)
            right_nonzero_indices = right_window.pixels_indices(nonzero)

            # Append the window nonzero pixel indices to the lists
            left_lane_indices.append(left_nonzero_indices)
            right_lane_indices.append(right_nonzero_indices)

            left_x = left_window.x
            right_x = right_window.x

        # Concatenate all the windows nonzero pixel indices
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        self._left_lane = Line.Line(nonzero[1][left_lane_indices],
                                    nonzero[0][left_lane_indices],
                                    self._height,
                                    self._width)

        self._right_lane = Line.Line(nonzero[1][right_lane_indices],
                                     nonzero[0][right_lane_indices],
                                     self._height,
                                     self._width)

    def find_lanes(self, frame, heatmap):
        # pipeline returns bird eye view perspective image, after applying sobel gradient
        # and color thresholds. The return image has pixel values of 0 and 1
        bird_eye_view = Pipeline.GradientWarpPipeline()(frame)
        # find all the nonzero pixels for the left lane and fit a polynomial curve
        self.find_left_lane(bird_eye_view)
        # find all the nonzero pixels for the right lane and fit a polynomial curve
        self.find_right_lane(bird_eye_view)

        frame = self.draw_processing_overlay(frame, heatmap)
        image = self.draw_lanes(frame, unwarp=True)
        return image

    def draw_processing_overlay(self, frame, heat_map=None):
        # Draw a overlay ribbon by increasing the intensity of the top 250 pixel rows
        # Overlay ribbon has 3 areas
        # 1. Windows overlay
        # 2. Bird eye view
        # 3. Curvature radius and camera distance
        frame[:250, :, :] = frame[:250, :, :] * .4
        # Draw windows overlay
        window_overlay = self.draw_windows(frame)
        window_overlay = cv2.resize(window_overlay, (0, 0), fx=0.3, fy=0.3)
        (h, w, _) = window_overlay.shape
        frame[20:20 + h, 20:20 + w, :] = window_overlay
        # Draw bird eye view
        #bird_eye_view = Pipeline.Warper()(frame)
        #bird_eye_view = self.draw_lanes(bird_eye_view)
        #bird_eye_view = cv2.resize(bird_eye_view, (0, 0), fx=0.3, fy=0.3)
        #frame[20:20 + h, 2 * 20 + w: 2 * (20 + w), :] = bird_eye_view
        heat_map_copy = np.copy(heat_map)
        #mask = heat_map_copy >= 1
        #heat_map_copy[mask] = 1
        heat_map_copy = np.dstack((heat_map_copy * 10, heat_map_copy * 10, heat_map_copy * 10))
        heat_map_copy = cv2.resize(heat_map_copy, (0, 0), fx=0.3, fy=0.3)
        frame[20:20 + h, 2 * 20 + w: 2 * (20 + w), :] = heat_map_copy
        # Draw curvature radius and camera distance
        text_x = 2 * (20 + w) + 20
        self.draw_text(frame, 'Radius of curvature:  {} m'.format(self.measure_curvature()), text_x, 80)
        self.draw_text(frame, 'Position (left):       {:.1f} m'.format(self._left_lane.vehicle_position()), text_x, 140)
        self.draw_text(frame, 'Position (right):      {:.1f} m'.format(self._right_lane.vehicle_position()), text_x, 200)
        return frame

    def draw_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    def find_left_lane(self, frame):
        self._current_left_x, self._current_left_y = \
            self.find_lane(frame, self._left_lane_windows)
        self._left_lane.fit(self._current_left_x, self._current_left_y)

    def find_right_lane(self, frame):
        self._current_right_x, self._current_right_y = \
            self.find_lane(frame, self._right_lane_windows)
        self._right_lane.fit(self._current_right_x, self._current_right_y)

    def find_lane(self, frame, windows):
        indices = []
        # Get nonzero pixels in the image. The nonzero() return a tuple of x, y, z points
        nonzero = frame.nonzero()
        x = None
        for window in windows:
            indices.append(window.pixels_indices(nonzero, x))
            x = window.mean_x
        indices = np.concatenate(indices)
        return nonzero[1][indices], nonzero[0][indices]

    def draw_lanes(self, frame, unwarp=False):
        image = np.zeros_like(frame).astype(np.uint8)
        points = np.vstack((self._left_lane.points(), np.flipud(self._right_lane.points())))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(image, [points], (0, 255, 0))
        if unwarp:
            image = Pipeline.Warper().unwarp(image)
        # Combine the result with the original image
        return cv2.addWeighted(frame, 1, image, 0.3, 0)

    def draw_windows(self, frame):
        image = Pipeline.GradientWarpPipeline()(frame, stacked=True)
        for window in self._left_lane_windows:
            coordinates = window.coordinates()
            cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)

        for window in self._right_lane_windows:
            coordinates = window.coordinates()
            cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)

        cv2.polylines(image, [self._left_lane.points()], False, (1., 0, 0), 2)
        cv2.polylines(image, [self._right_lane.points()], False, (1., 0, 0), 2)
        return image * 255

    def measure_curvature(self):
        return int(np.average([self._left_lane.measure_curvature(),
                               self._right_lane.measure_curvature()]))
