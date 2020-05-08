'''
Lets make a SICK numpy library to replace PIL

Goals:
-Use [0,1] range for all pixels, not [0, 255]
-Do not ever engage PIL, we want to deal entirely with np.arrays here
-Draw with rectangles and be able to blend with transparency
-Compute cost adjustment
-Build out hill climbing algorithm
'''

import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


IMAGE_DEFAULT_CHANNELS = 4


class ImageEvolve:
    def __init__(self, url):
        im = Image.open(url)
        self.goal_array = np.array(im) / 255.0
        self.goal_dim = self.goal_array.shape  # We do not care how many channels
        self.candidate_array = np.ones(shape=self.goal_dim)
        self.squared_error = self.compute_squared_error([0, 0, self.goal_dim[0], self.goal_dim[1]])

    def clip(self, val, bound):
        return min(bound, max(0, val))

    def random_rectangle(self):
        x_center = random.randint(0, self.goal_dim[0])
        y_center = random.randint(0, self.goal_dim[1])
        x_radius = random.randint(3, 16)
        y_radius = random.randint(3, 16)
        color = tuple([random.random() for i in range(3)])  # create an RGB color
        return {
            'bounding_box': (
                self.clip(x_center - x_radius, self.goal_dim[0]),
                self.clip(y_center - y_radius, self.goal_dim[1]),
                self.clip(x_center + x_radius, self.goal_dim[0]),
                self.clip(y_center + y_radius, self.goal_dim[1])
            ),
            'color': color,
            'transparency': random.random()
        }

    def compute_squared_error(self, bound):
        candidate_region = self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :]
        goal_region = self.goal_array[bound[0]: bound[2], bound[1]: bound[3], :]
        return np.sum((candidate_region - goal_region)**2)

    def apply_rectangle(self, bound, shape_filter, transparency):
        self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :] *= 1.0 - transparency
        self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :] += transparency * shape_filter

    def perform_hill_climb_iteration(self):
        rect = self.random_rectangle()
        bound = rect['bounding_box']
        x_dim = bound[2] - bound[0]
        y_dim = bound[3] - bound[1]
        shape_filter = np.ones(shape=(x_dim, y_dim, 3))
        for channel in range(3):
            shape_filter[:, :, channel] *= rect['color'][channel]
        region_before = np.copy(self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :])
        sum_se_before = self.compute_squared_error(rect['bounding_box'])
        self.apply_rectangle(bound, shape_filter, rect['transparency'])
        sum_se_after = self.compute_squared_error(rect['bounding_box'])
        if sum_se_after < sum_se_before:  # We improved our approximation!
            self.squared_error -= (sum_se_before - sum_se_after)
            return True
        self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :] = region_before
        return False

    def adapt_image(self, iterations):
        costs = [self.squared_error]
        for i in range(iterations):
            if self.perform_hill_climb_iteration():  # This means we successfully added a shape!
                costs.append(self.squared_error)
            if not i % max(1, int(iterations * 0.05)):
                print('Iteration:', i, '--', len(costs) - 1, 'added shapes.')
        plt.imshow(self.candidate_array)
        plt.show()
        plt.plot(costs)
        plt.show()


evolve = ImageEvolve('background_anchor_large_2x-610x302.jpg')
evolve.adapt_image(100000)
