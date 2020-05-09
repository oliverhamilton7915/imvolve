import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os


class ImageEvolve:
    def __init__(self, url):
        # Open and parse goal image
        self.source_url = url
        im = Image.open(self.source_url)
        self.goal_array = np.array(im) / 255.0
        self.goal_dim = self.goal_array.shape  # We do not care how many channels

        # Create candidate and set configs
        self.candidate_array = np.ones(shape=self.goal_dim)
        self.squared_error = self.compute_squared_error([0, 0, self.goal_dim[0], self.goal_dim[1]])
        self.snapshot_file_names = []
        self.snapshot_dir = './snapshots/'
        self.gif_dir = './gifs/'
        self.rec_config = {
            'radius_lower_bound': 50,
            'radius_upper_bound': 200,
        }

    # Use self.rec_config to generate a random rectangle
    def random_rectangle(self):

        x_center = random.randint(0, self.goal_dim[0])
        y_center = random.randint(0, self.goal_dim[1])
        x_radius = random.randint(self.rec_config['radius_lower_bound'], self.rec_config['radius_upper_bound'])
        y_radius = random.randint(self.rec_config['radius_lower_bound'], self.rec_config['radius_upper_bound'])
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
        return np.sum((candidate_region - goal_region)**2)

    # This method applies a rectangle mask over the candidate image
    def apply_rectangle(self, bound, shape_filter, transparency):
        self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :] *= 1.0 - transparency
        self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :] += transparency * shape_filter

    # This method picks a rectangle, adds it over the image, and backtracks if the image is left worse off
    def perform_hill_climb_iteration(self):
        rect = self.random_rectangle()
        bound = rect['bounding_box']

        # Create the rectangle filter
        x_dim = bound[2] - bound[0]  # compute the filter width from the bounding box
        y_dim = bound[3] - bound[1]  # compute the filter height from the bounding box
        shape_filter = np.ones(shape=(x_dim, y_dim, 3))
        for channel in range(3):
            shape_filter[:, :, channel] *= rect['color'][channel]

        # Before applying filter, assess the squared error over the region
        region_before = np.copy(self.candidate_array[bound[0]: bound[2], bound[1]: bound[3], :])
        sum_se_before = self.compute_squared_error(rect['bounding_box'])

        # Apply the rectangle filter and compute the new squared error
        self.apply_rectangle(bound, shape_filter, rect['transparency'])
        sum_se_after = self.compute_squared_error(rect['bounding_box'])

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
            if self.perform_hill_climb_iteration():  # This means we successfully added a shape!
                costs.append(self.squared_error)
            if not i % max(1, int(iterations * 0.05)):
                print('Iteration:', i, '--', len(costs) - 1, 'added shapes.')
                file_name = 'snapshot_iter_' + str(i) + '.png'
                self.save_image(file_name)
                self.snapshot_file_names.append(file_name)

        self.compile_gif_and_save(self.source_url + '_' + str(iterations) + '_iterations.gif')
        plt.plot(costs)
        plt.show()

    # This method appends file name to the appropriate project directory and saves a temporary image snapshot
    def save_image(self, name):
        proxy_image = (255.0 * self.candidate_array).astype(np.uint8)
        proxy_image = Image.fromarray(proxy_image)
        proxy_image.save(self.snapshot_dir + name)
        print('Image snapshot saved to: ' + self.snapshot_dir + name)

    # This method deletes all saved image snapshots and creates a gif out of them
    def compile_gif_and_save(self, name):
        images = []
        for filename in self.snapshot_file_names:
            images.append(imageio.imread(self.snapshot_dir + filename))
            os.remove(self.snapshot_dir + filename)  # Delete the saved image once it finds its home in the GIF
        imageio.mimsave(self.gif_dir + name, images)  # Save the GIF


if __name__ == '__main__':
    evolve = ImageEvolve('large_demo_2880x1490.jpg')
    evolve.adapt_image(10000)
