import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
from copy import deepcopy


def compute_fitness(candidate, goal):
    assert(candidate.shape == goal.shape)
    return np.sum((goal - candidate)**2)  # MSE


def show_image(pix_array, print_shape = False):
    if print_shape:
        print("Shape of the pixel array is: ", pix_array.shape)
    plt.imshow(pix_array)
    plt.show()


def initialize_shapes(shapes_arr, num_shapes, canvas_shape):
    for i in range(num_shapes):
        x = random.randint(0, canvas_shape[0] - 1)
        y = random.randint(0, canvas_shape[1] - 1)
        r = random.randint(5, 50)
        chan_vals = [random.randint(0, 255) for i in range(4)]
        shapes_arr.append([x, y, r] + chan_vals)


def render_shapes(shapes_arr, canvas_shape):
    im = Image.new('RGB', canvas_shape[:-1], (255, 255, 255))
    draw_candidate = ImageDraw.Draw(im, 'RGBA')
    for (x, y, r, R, G, B, A) in shapes_arr:
        draw_candidate.ellipse([x - r, y - r, x + r, y + r], fill=(R, G, B, A))
    del draw_candidate
    return np.array(im)


def make_mutation(shapes_arr, goal_shape):
    gene_index = random.randint(0, len(shapes_arr) - 1)
    position_index = random.randint(0, len(shapes_arr[gene_index]) - 1)
    delta = random.choice([1, -1, 3, -3, 5, -5])
    val = shapes_arr[gene_index][position_index] + delta
    if position_index > 2:  # color change
        shapes_arr[gene_index][position_index] = max(0, min(val, 255))
    elif position_index == 0:  # x-coordinate change
        shapes_arr[gene_index][position_index] = max(0, min(goal_shape[0], val))
    elif position_index == 1:  # y-coordinate change
        shapes_arr[gene_index][position_index] = max(0, min(goal_shape[1], val))
    return shapes_arr


def _make_mutation(pixels_arr):
    im = Image.fromarray(pixels_arr)
    im_draw = ImageDraw.Draw(im)
    radius = random.randint(3,10)
    x = random.randint(0, pixels_arr.shape[0])
    y = random.randint(0, pixels_arr.shape[1])
    colors = tuple([random.randint(0, 255) for i in range(4)])
    im_draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=colors)
    return np.array(im)


if __name__ == '__main__':
    # 1. load up goal image
    goal_image = Image.open('brand.jpg')  # a small picture of the firefox logo
    goal_pixels = np.array(goal_image)  # convert to numpy array

    # 2. randomly place shapes on candidate image
    '''
    candidate_shapes = []  # each shape is going to be encoded as something like: [23, 212, 27, 109, 34, 65, 188]
                           # the first 3 values denote (x-coord, y-coord, radius)
                           # the last 4 values denote the color (RGB) + transparency of the circle
    initialize_shapes(candidate_shapes, 0, goal_pixels.shape)
    '''
    candidate_pixels = render_shapes([], goal_pixels.shape)  # initially we render no shapes.

    # 3. performs hill climbing + track cost decline
    num_added = 0
    costs = list()
    costs.append(compute_fitness(candidate_pixels, goal_pixels))
    num_iterations = 10000
    for k in range(1, num_iterations):
        neighbor_pixels = _make_mutation(candidate_pixels)
        neighbor_cost = compute_fitness(neighbor_pixels, goal_pixels)
        if neighbor_cost < costs[-1]:
            num_added += 1
            costs.append(neighbor_cost)
            candidate_pixels = neighbor_pixels
        if not k % 1000:
            print('Iteration:', k, '--', num_added, 'added mutations')

    # 4. show images
    show_image(goal_pixels, True)
    show_image(candidate_pixels, True)
    plt.plot(costs)
    plt.title('Costs')
    plt.show()


