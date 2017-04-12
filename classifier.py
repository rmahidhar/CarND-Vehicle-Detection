import matplotlib.image
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from features import FeatureVector
import glob
import numpy as np


class Classifier(object):
    def __init__(self, cars_files=None, non_cars_files=None):

        if cars_files:
            self._cars_files = cars_files
        else:
            self._cars_files = glob.glob('data/vehicles/*/*.png')
        if non_cars_files:
            self._non_cars_files = non_cars_files
        else:
            self._non_cars_files = glob.glob('data/non-vehicles/*/*.png')
        self._scaler = None
        self._svc = None
        self.fit()

    def fit(self):
        cars = []
        non_cars = []

        print('Reading car images')
        # Read car and non car images
        for file in self._cars_files:
            cars.append(matplotlib.image.imread(file))

        print('Reading non car images')
        for file in self._non_cars_files:
            non_cars.append(matplotlib.image.imread(file))

        cars = np.asarray(cars)
        non_cars = np.asarray(non_cars)

        print('Vehicles images:{}'.format(cars.shape))
        print('Non-vehicles images:{}'.format(non_cars.shape))

        # extract features from car and non-car images
        car_features = []
        non_car_features = []

        print('Extracting cars image features')
        for car in cars:
            car_features.append(FeatureVector(car)())

        print('Extracting non car image features')
        for non_car in non_cars:
            non_car_features.append(FeatureVector(non_car)())

        X = np.vstack((car_features, non_car_features)).astype(np.float64)

        print('Scaling combined car and non car features')
        # scale the features
        self._scaler = StandardScaler().fit(X)
        scaled_X = self._scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

        print('Training SVM classifer')
        # train the classifier
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)
        self._svc = LinearSVC()
        self._svc.fit(X_train, y_train)
        accuracy = round(self._svc.score(X_test, y_test), 4)
        print("Accuracy:{}".format(accuracy))

    def __call__(self, features):
        features = self._scaler.transform(np.array(features).reshape(1, -1))
        prediction = self._svc.predict(features)
        return prediction