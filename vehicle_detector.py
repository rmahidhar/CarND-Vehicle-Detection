from features import FeatureExtractor
import numpy as np
import collections
from scipy.ndimage.measurements import label
import cv2
from LaneFinder import LaneFinder


class VehicleDetector(object):
    def __init__(self, image, classifier):
        self._image = image
        self._classifier = classifier
        self._y_start = 400
        self._y_stop = 656
        self._x_start = image.shape[1] // 2
        self._x_stop = image.shape[1]
        self._scales = [1.0, 1.5, 2.0]
        self._detections = np.empty([0, 4], dtype=np.int64)
        self._previous_detections = collections.deque(maxlen=20)
        self._plotter = None
        self._lane_finder = LaneFinder(image)
        self._heat_map = None

    def detect_vehicles(self, image):
        self._image = image
        detections = np.empty([0, 4], dtype=np.int64)
        for scale in self._scales:
            detections = np.append(detections, self.scan_image(scale), axis=0)

        self._detections = detections

        detections, self._heat_map = self.merge_detections(detections)

        self._previous_detections.append(detections)

        detections, _ = self.merge_detections(
            np.concatenate(np.array(self._previous_detections)), threshold=min(len(self._previous_detections), 15)
        )

        for c in detections:
            cv2.rectangle(image, (c[0], c[1]), (c[2], c[3]), (0, 0, 255), 3)

        return self._lane_finder.find_lanes(image, self._heat_map)

    def heat_map(self):
        return self._heat_map

    def detections(self):
        return self._detections

    def scan_image(self, scale):
        extractor = FeatureExtractor(self._image,
                                     (self._x_start, self._x_stop),
                                     (self._y_start, self._y_stop),
                                     scale=scale)
        detections = np.empty([0, 4], dtype=np.int64)
        for features in extractor():
            prediction = self._classifier(features.feature_vector)
            if prediction == 1:
                x_left = np.int(features.x_left * scale)
                y_top = np.int(features.y_top * scale)
                win_size = np.int(64 * scale)
                detections = np.append(detections, [[x_left,
                                                     y_top + self._y_start,
                                                     x_left + win_size,
                                                     y_top + win_size + self._y_start]], axis=0)
        return detections

    def merge_detections(self, detections, threshold=1):
        # Add heat to each box in box list
        heat_map = self.compute_heatmap(detections)
        # Apply threshold to help remove false positives
        heat_map[heat_map < threshold] = 0
        heat_map = np.clip(heat_map, 0, 255)
        labels = label(heat_map)
        cars = np.empty([0, 4], dtype=np.int64)
        # Iterate through all detected cars
        for car in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car).nonzero()
            cars = np.append(
                cars,
                [[np.min(nonzero[1]), np.min(nonzero[0]), np.max(nonzero[1]), np.max(nonzero[0])]],
                axis=0
            )
        # Return the image
        return cars, heat_map

    def compute_heatmap(self, regions):
        heat_map = np.zeros_like(self._image[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        for region in regions:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heat_map[region[1]:region[3], region[0]:region[2]] += 1
        return heat_map
