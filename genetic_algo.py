import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


class GeneticImage:
    def __init__(self, url):
        # Establish config
        self.config = {
            'radius_lower_bound': 5,
            'radius_upper_bound': 20,
            'starting_shapes': 100
        }

        # Open and parse goal image
        self.source_url = url
        im = Image.open(self.source_url)
        self.goal_array = np.array(im) / 255.0
        self.goal_dim = self.goal_array.shape

        # Create candidate image
        self.candidate_array = np.ones(shape=self.goal_dim)
        self.rendered_shapes = []
        for shape in range(self.config['starting_shapes']):
            self.apply_rectangle(self.random_rectangle())
        self.squared_error = self.compute_squared_error([0, 0, self.goal_dim[0], self.goal_dim[1]])

        # Other
        self.gif_dir = './gifs/'
        self.image_cache = []

    # Use self.rec_config to generate a random rectangle
    def random_rectangle(self):
        x_center = random.randint(0, self.goal_dim[0])
        y_center = random.randint(0, self.goal_dim[1])
        x_radius = random.randint(self.config['radius_lower_bound'], self.config['radius_upper_bound'])
        y_radius = random.randint(self.config['radius_lower_bound'], self.config['radius_upper_bound'])
        color = tuple([random.random() for i in range(3)])  # create an RGB color

        return {
            'bounding_box': (
                min(max(0, x_center - x_radius), self.goal_dim[0]),
                min(max(0, y_center - y_radius), self.goal_dim[1]),
                min(max(0, x_center + x_radius), self.goal_dim[0]),
                min(max(0, y_center + y_radius), self.goal_dim[1])
            ),
            'color': color,
            'transparency': random.random()
        }

    # This method computes the squared error across this supplied bound
    # bound = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    def compute_squared_error(self, bound):
        candidate_region = self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :]
        goal_region = self.goal_array[bound[0]: bound[2], bound[1]: bound[3], :]
        return np.sum((candidate_region - goal_region) ** 2)

    # This method applies a rectangle mask over the candidate image
    def apply_rectangle(self, rect):
        bound = rect['bounding_box']

        # Create the rectangle filter
        x_dim = bound[2] - bound[0]  # compute the filter width from the bounding box
        y_dim = bound[3] - bound[1]  # compute the filter height from the bounding box
        shape_filter = np.ones(shape=(x_dim, y_dim, 3))
        for channel in range(3):
            shape_filter[:, :, channel] *= rect['color'][channel]

        self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :] *= 1.0 - rect['transparency']
        self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :] += rect['transparency'] * shape_filter

    '''
    1. We need to write a function to "un-render" shape filters
    2. We need to write a function to generate random mutations to existing shapes
    3. We need to write a function that generates a random mutation, updates the image, and recomputes cost
    4. Lastly, we need to adjust the function below to mutate some of the shapes at each iteration
    '''

    # This method picks a rectangle, adds it over the image, and backtracks if the image is left worse off
    def perform_mutation_iteration(self):
        shape = self.random_rectangle()
        bound = shape['bounding_box']

        # Before applying filter, assess the squared error over the region
        region_before = np.copy(self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :])
        sum_se_before = self.compute_squared_error(shape['bounding_box'])

        # Apply the rectangle filter and compute the new squared error
        self.apply_rectangle(shape)
        sum_se_after = self.compute_squared_error(shape['bounding_box'])

        # Test if the added rectangle 'helped' and return early if so
        if sum_se_after < sum_se_before:  # We improved our approximation!
            self.squared_error -= (sum_se_before - sum_se_after)
            return True

        # Else, put back the original background pixels
        self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :] = region_before
        return False

    # For a certain number of iterations
    def adapt_image(self, iterations):
        costs = [self.squared_error]  # this is the starting 'fitness' of our candidate image

        for i in range(iterations):
            if self.perform_mutation_iteration():  # This means we successfully added a shape!
                costs.append(self.squared_error)
            if not i % max(1, int(iterations * 0.05)):
                print('Iteration:', i, '--', len(costs) - 1, 'added shapes.')
                self.cache_image()

        self.compile_gif_and_save(self.source_url + '_' + str(iterations) + '_iterations.gif')
        plt.plot(costs)
        plt.show()

    # This method caches an array of pixel values in the `image_cache` state variable
    def cache_image(self):
        proxy_image = (255.0 * self.candidate_array).astype(np.uint8)
        self.image_cache.append(proxy_image)

    # This method retrieves all cached images and creates a gif out of them
    def compile_gif_and_save(self, name):
        images = [Image.fromarray(im) for im in self.image_cache]  # generate image for each cached array
        imageio.mimsave(self.gif_dir + name, images)  # save images as gif


if __name__ == '__main__':
    ga = GeneticImage('medium_demo_832x468.jpg')
    ga.adapt_image(100)