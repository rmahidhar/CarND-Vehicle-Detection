import math
import matplotlib.pyplot as plt


class ImagePlotter(object):
    def __init__(self, num_images=1, grid=(None, None), figsize=(10, 5), title=None):
        self.num_images = num_images
        if grid[0] is None or grid[1] is None:
            self.rows, self.cols = self.get_grid_dim(num_images)
        else:
            self.rows, self.cols = grid[0], grid[1]

        self.fig, self.axes = plt.subplots(self.rows, self.cols, figsize=figsize)
        if self.num_images > 1:
            self.axes = self.axes.ravel()

        #if title:
        #    self.fig.suptitle(title)

        #self.fig.tight_layout()
        self.image_num = 0

    def get_grid_dim(self, x):
        """
        Transforms x into product of two integers
        """
        factors = self.prime_powers(x)
        if len(factors) % 2 == 0:
            i = int(len(factors) / 2)
            return factors[i], factors[i - 1]

        i = len(factors) // 2
        return factors[i], factors[i]

    def prime_powers(self, n):
        """
        Compute the factors of a positive integer
        Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
        :param n: int
        :return: set
        """
        factors = set()
        for x in range(1, int(math.sqrt(n)) + 1):
            if n % x == 0:
                factors.add(int(x))
                factors.add(int(n // x))
        return sorted(factors)

    def __call__(self, image, title=None, axis='off'):
        if self.num_images == 1:
            axes = self.axes
        else:
            axes = self.axes[self.image_num]
            self.image_num += 1
        axes.imshow(image)
        axes.axis(axis)
        if title:
            axes.set_title(title)
