
import numpy as np
import collections


class Line(object):
    def __init__(self, x, y, height, width):
        self._height = height
        self._width = width
        self._recent_fitted_x = collections.deque(maxlen=5)
        self._recent_fits = collections.deque(maxlen=5)
        self._current_fit = None

        self._init(x, y)

    def _init(self, x, y):
        self.fit(x, y)

    def fit(self, x, y):
        if len(y) > 0 and \
           (self._current_fit is None or np.max(y) - np.min(y) > self._height * .625):
            self._current_fit = np.polyfit(y, x, 2)
            self._recent_fits.append(self._current_fit)
            self._recent_fitted_x.append(x)

    def points(self):
        plot_y = np.linspace(0, self._height-1, self._height)
        best_fit = np.array(self._recent_fits).mean(axis=0)
        best_fit_x = best_fit[0] * plot_y ** 2 + best_fit[1] * plot_y + best_fit[2]
        return np.stack((best_fit_x, plot_y)).astype(int).T

    def windows(self, margin):
        plot_y = np.linspace(0, self._height-1, self._height)
        best_fit = np.array(self._recent_fits).mean(axis=0)
        best_fit_x = best_fit[0] * plot_y ** 2 + best_fit[1] * plot_y + best_fit[2]

        line_window1 = np.array([np.transpose(np.vstack([[best_fit_x - margin, plot_y]]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([[best_fit_x + margin, plot_y]])))])
        return np.hstack((line_window1, line_window2))

    def measure_curvature(self):
        points = self.points()
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720  # meters per pixel in y dimension
        xm_per_pix = 3.7/700  # meters per pixel in x dimension

        x = points[:, 0]
        y = points[:, 1]

        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        curve_radius = ((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2 ) ** 1.5) \
                / np.absolute(2 * fit_cr[0])
        return int(curve_radius)

    def vehicle_position(self):
        points = self.points()
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        x = points[np.max(points[:, 1])][0]
        return np.absolute((self._width // 2 - x) * xm_per_pix)