import cv2
import numpy as np
import skimage.feature
import skimage.transform
import matplotlib.image


class WindowFeatures(object):
    def __init__(self, features, x_left, y_top):
        self.feature_vector = features
        self.x_left = x_left
        self.y_top = y_top


class FeatureVector(object):
    def __init__(self, image):
        self._image = image
        self._image = cv2.cvtColor(self._image, cv2.COLOR_RGB2YCrCb)

    def __call__(self):
        image = self._image
        # hog features
        hog_features = []
        for channel in range(image.shape[2]):
            hog_features.append(
                skimage.feature.hog(image[:, :, channel], orientations=10, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), transform_sqrt=False,
                                    visualise=False, feature_vector=True)
            )
        hog_features = np.concatenate(hog_features)

        # histogram features
        n_bins = 32
        bins_range = (0, 256)
        channel1_hist = np.histogram(image[:, :, 0], bins=n_bins, range=bins_range)
        channel2_hist = np.histogram(image[:, :, 1], bins=n_bins, range=bins_range)
        channel3_hist = np.histogram(image[:, :, 2], bins=n_bins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

        # spatial_binned features
        color1 = cv2.resize(image[:, :, 0], (32, 32)).ravel()
        color2 = cv2.resize(image[:, :, 1], (32, 32)).ravel()
        color3 = cv2.resize(image[:, :, 2], (32, 32)).ravel()
        spatial_features = np.hstack((color1, color2, color3))

        return np.concatenate((spatial_features, hist_features, hog_features))


class FeatureExtractor(object):
    def __init__(self, image, x_region, y_region, window_size=64,
                 scale=1, orientations=10, pixels_per_cell=8, cells_per_block=2):
        self._image = image
        self._orientations = orientations
        self._pixels_per_cell = (pixels_per_cell, pixels_per_cell)
        self._cells_per_block = (cells_per_block, cells_per_block)
        self._x_start = x_region[0]
        self._x_stop = x_region[1]
        self._y_start = y_region[0]
        self._y_stop = y_region[1]
        self._scale = scale
        self._window_size = window_size
        self._cells_per_step = 2
        self._blocks_per_window = (self._window_size // self._pixels_per_cell[0]) - 1

    def hog(self, image, visualise=False, feature_vector=False):
        features, hog_image = skimage.feature.hog(image,
                                                  orientations=self._orientations,
                                                  pixels_per_cell=self._pixels_per_cell,
                                                  cells_per_block=self._cells_per_block,
                                                  transform_sqrt=False,
                                                  visualise=True,
                                                  feature_vector=feature_vector)
        if visualise:
            return features, hog_image
        else:
            return features

    def spatial_binned(self, image, color_space='RGB', size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector
        color1 = cv2.resize(image[:, :, 0], size).ravel()
        color2 = cv2.resize(image[:, :, 1], size).ravel()
        color3 = cv2.resize(image[:, :, 2], size).ravel()
        return np.hstack((color1, color2, color3))

    # compute color histogram features
    def color_hist(self, image, nbins=32, bins_range=(0, 256)):
        # np.histogram() returns a tuple of two arrays. item one contains the counts in each of the bins
        # and other contains the bin edges( it will be one element longer than the item one)
        color1 = np.histogram(image[:, :, 0], bins=nbins, range=bins_range)
        color2 = np.histogram(image[:, :, 1], bins=nbins, range=bins_range)
        color3 = np.histogram(image[:, :, 2], bins=nbins, range=bins_range)
        return np.concatenate((color1[0], color2[0], color3[0]))

    def image_grid(self, h, w):
        pix_per_cell = self._pixels_per_cell[0]
        n_col_blocks = (w // pix_per_cell) - 1
        n_row_blocks = (h // pix_per_cell) - 1
        cols = (n_col_blocks - self._blocks_per_window) // self._cells_per_step
        rows = (n_row_blocks - self._blocks_per_window) // self._cells_per_step
        return rows, cols

    def __call__(self):
        img = self._image.astype(np.float32) / 255

        # Bound the image search
        img_to_search = img[self._y_start:self._y_stop, :, :]
        img_to_search = self.convert_color(img_to_search, conv='RGB2YCrCb')
        if self._scale != 1:
            img_shape = img_to_search.shape
            img_to_search = cv2.resize(img_to_search,
                                       (np.int(img_shape[1] / self._scale),
                                        np.int(img_shape[0] / self._scale)))

        ch1 = img_to_search[:, :, 0]
        ch2 = img_to_search[:, :, 1]
        ch3 = img_to_search[:, :, 2]

        rows, cols = self.image_grid(ch1.shape[0], ch1.shape[1])

        # Compute individual channel HOG features for the entire image
        hog_red = self.hog(ch1)
        hog_green = self.hog(ch2)
        hog_blue = self.hog(ch3)

        for xb in range(cols):
            for yb in range(rows):
                y_pos = yb * self._cells_per_step
                x_pos = xb * self._cells_per_step

                # Extract HOG features for this patch
                hog_feat_1 = hog_red[y_pos:y_pos+self._blocks_per_window, x_pos:x_pos+self._blocks_per_window].ravel()
                hog_feat_2 = hog_green[y_pos:y_pos+self._blocks_per_window, x_pos:x_pos+self._blocks_per_window].ravel()
                hog_feat_3 = hog_blue[y_pos:y_pos+self._blocks_per_window, x_pos:x_pos+self._blocks_per_window].ravel()
                hog_features = np.hstack((hog_feat_1, hog_feat_2, hog_feat_3))

                x_left = x_pos * self._pixels_per_cell[0]
                y_top = y_pos * self._pixels_per_cell[0]

                # Extract the image patch
                sub_img = cv2.resize(img_to_search[y_top:y_top + self._window_size, x_left:x_left + self._window_size],
                                     (64, 64))
                # Get color features
                spatial_features = self.spatial_binned(sub_img, size=(32, 32))
                # Get histogram features
                hist_features = self.color_hist(sub_img, nbins=32)
                #print("spatial:{}, hist:{}, hog:{}".format(spatial_features.shape, hist_features.shape, hog_features.shape))
                features = np.concatenate((spatial_features, hist_features, hog_features))
                yield WindowFeatures(features, x_left, y_top)

    def draw_features_windows(self):
        image = self._image.astype(np.float32)
        img_to_search = image[self._y_start:self._y_stop, :, :]
        if self._scale != 1:
            img_shape = img_to_search.shape
            img_to_search = cv2.resize(img_to_search,
                                       (np.int(img_shape[1] / self._scale),
                                        np.int(img_shape[0] / self._scale)))

        ch1 = img_to_search[:, :, 0]

        rows, cols = self.image_grid(ch1.shape[0], ch1.shape[1])

        for xb in range(cols):
            for yb in range(rows):
                y_pos = yb * self._cells_per_step
                x_pos = xb * self._cells_per_step

                x_left = x_pos * self._pixels_per_cell[0]
                y_top = y_pos * self._pixels_per_cell[0]

                x_left = np.int(x_left * self._scale)
                y_top = np.int(y_top * self._scale)
                win_size = np.int(self._window_size * self._scale)

                #cv2.rectangle(image, (x_left, y_top + self._y_start),
                #              (x_left + self._window_size, y_top + self._window_size + self._y_start), (0, 0, 255), 2)

                cv2.rectangle(self._image, (x_left, y_top + self._y_start),
                              (x_left + win_size, y_top + win_size + self._y_start), (0, 0, 255), 2)

        return self._image

    def convert_color(self, img, conv='RGB2YCrCb'):
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'RGB2LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
