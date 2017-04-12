
##**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./images/car_images.png
[image2]: ./images/hog.png
[image3]: ./images/scale.png
[image4]: ./images/scale1.png
[image5]: ./images/scale2.png
[image6]: ./images/windows.png
[image7]: ./images/heatmap.png
[image8]: ./images/detections.png
[video1]: ./images/project_video.gif

### Feature Extraction

Features describe the characteristics of an object, and with images, it really all comes down to intensity and gradients of intensity, and how these features capture the color and shape of an object. We're interested in capturing color and shape information, and with various combinations of pixel intensities, histograms, and gradients, we can accomplish that. After experimenting with various features I settled on a combination of HOG (Histogram of Oriented Gradients), spatial information and color channel histograms, all using YCbCr color space.

####Histogram of Oriented Gradients (HOG) Features

I used the parameters given in the course videos. I used the following HOG parameters used for the vehicle detections are

parameter | value 
:---: | :---: 
orientation | 9 
pixels per cell | (8, 8)
cells per block | (2, 2)
 
Dataset Random Car Images 

![alt text][image1]

HOG on Random Car Images

![alt text][image2]

The HOG for the image is computed using the hog() function in skimage.feature. The HOG is computed for each channel independently and then concatenated. During sliding window processing subset(size of window) of the hog features are extracted from the precomputed HOG features on the entire image.

```python
        for channel in range(image.shape[2]):
            hog_features.append(
                skimage.feature.hog(image[:, :, channel], orientations=10, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), transform_sqrt=False,
                                    visualise=False, feature_vector=True)
            )
        hog_features = np.concatenate(hog_features)
```

#### Color channel histogram Features

The color channel histogram is computed on individual channel breaking into 32 bins within (0, 256) range and is included in the features set. The color channel histogram is computed on sub image equal to the size of the window size and resize to (64, 64) which is the dimension of training and test images. 

```python
        n_bins = 32
        bins_range = (0, 256)
        channel1_hist = np.histogram(image[:, :, 0], bins=n_bins, range=bins_range)
        channel2_hist = np.histogram(image[:, :, 1], bins=n_bins, range=bins_range)
        channel3_hist = np.histogram(image[:, :, 2], bins=n_bins, range=bins_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
```

#### Spatial Binned Features

The Spatial information is computed on individual channels resizing the image to (32, 32) and flatten to 1-D vector. The Spatial information is also extracted from the sub image of size equal to the sliding window size. 

```python
        size = (32, 32)
        color1 = cv2.resize(image[:, :, 0], size).ravel()
        color2 = cv2.resize(image[:, :, 1], size).ravel()
        color3 = cv2.resize(image[:, :, 2], size).ravel()
        np.hstack((color1, color2, color3))
```

All the above features are combined in the feature extractor generator class 

```python
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
```

### Traning a classifer

Trained Support Vector Machine classifier from sklearn library using the aforementioned feature extractor.

```python
        X = np.vstack((car_features, non_car_features)).astype(np.float64)

        # scale the features
        self._scaler = StandardScaler().fit(X)
        scaled_X = self._scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

        # train the classifier
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)
        self._svc = LinearSVC()
        self._svc.fit(X_train, y_train)
        accuracy = round(self._svc.score(X_test, y_test), 4)
```

###Sliding Window Search

The classifier trained above is used in sliding window scan on the images to detect cars as follows. 

* Each frame is cropped vertically such that ROI is from the dashboard camera to the horizon. This avoids false detections in the upper half of the image with sky and trees. 
* The classifier is moved over the cropped image in sliding window fashion with overlap of 2px. This process is repeated by scaling the cropped image by 1.0, 1.5 and 2.0.

```python
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
        for region in detections:
            cv2.rectangle(image, (region[0], region[1]), (region[2], region[3]), (0, 0, 255), 3)
```

![alt_text][image3]
![alt_text][image4]
![alt_text][image5]

I chosen the scales based on the scale value used in course quiz. In Hog Sampling search quiz scale value of 1.5 is used and from this value i derived two more scale(1 and 2) that are +/- 0.5 of 1.5. The experiments with this scale gave decent results.

The following image shows predictions as returned by the our trained SVM classifier on the sliding window image patch.

![alt_text][image6]

For each prediction, 1 is added onto a heatmap in the area of the bounding box. 

![alt_text][image7]

Then the predictions are merged and labeled using the scipy.ndimage.measurements.label(). We still see the false positives and these will be removed using thresholding in the video processing.

![alt_text][image8]

---

### Video Implementation

A well trained classifier is moved over the image in sliding window fashion. I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used scipy.ndimage.measurements.label() to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.

<p align="center">
  <img src="images/project_video.gif" alt="Project Video"/>
</p>
 
Here's a [link to my video result](./project_video_annotated_vehicles.mp4)

---

###Discussion

I had to experiment with various thresholding valies for removing the false positives. After multiple experiments I settled down to thresholding value of 15. The video processing took 3 hours to finish and it convinced me this is not a pratical solution for the self driving cars. I'm sure there are better and faster approaches used in self driving cars for the vehicle detection which i will explore further.